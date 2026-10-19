"""Uncut FK playback with actual finger joints and explicit search-intent labels.

Can read completed checkpoint chunks while a native experiment is running.
It never steps that experiment, edits its state, or treats planned targets as
contacts. Prefix videos carry DEVELOPMENT PREFIX in every frame and metadata.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import zipfile

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Module.fr3_leap import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from Module.fr3_leap.model import compile_model_xml


KEYS = ('time_s', 'q_rad', 'tip_positions_world_m', 'contact_positions_world_m',
        'forces_n', 'active', 'phase', 'finger_search_active', 'finger_search_state',
        'finger_search_target_positions_world_m', 'finger_search_target_world_m',
        'finger_search_target_valid', 'coordinated_recovery_finger',
        'coordinated_recovery_sequence', 'coordinated_recovery_stage')
COLORS = ((.15, .65, 1., 1.), (.95, .25, .85, 1.), (.12, .85, .35, 1.), (1., .25, .25, 1.))
NAMES = ('INDEX', 'MIDDLE', 'RING', 'THUMB')


def load_recorded_trace(directory, *, through_s=None):
    directory = Path(directory)
    final = directory / 'trace.npz'
    files = [final] if final.exists() else sorted((directory / 'trace_chunks').glob('*.npz'))
    chunks, sources = [], []
    for path in files:
        raw = path.read_bytes()
        try:
            with np.load(BytesIO(raw), allow_pickle=False) as archive:
                stored_times = archive['time_s']
                count = len(stored_times) if through_s is None else int(np.searchsorted(stored_times,through_s+1e-9,side='right'))
                values = {key: archive[key][:count].copy() for key in KEYS if key in archive.files}
        except (EOFError, zipfile.BadZipFile):
            if path == files[-1] and path != final:
                break  # A concurrently written last chunk is not completed evidence.
            raise
        if not len(values.get('time_s', [])):
            continue
        chunks.append(values)
        sources.append(dict(file=str(path.resolve()), sha256=sha256(raw).hexdigest(),
                            start_s=float(values['time_s'][0]), end_s=float(values['time_s'][-1]),
                            stored_end_s=float(stored_times[-1]),
                            samples=len(values['time_s'])))
        if through_s is not None and stored_times[-1] >= through_s-1e-9:
            break
    if not chunks:
        raise ValueError('No completed native trace or checkpoint chunk found')
    keys = set(chunks[0])
    if any(set(chunk) != keys for chunk in chunks):
        raise ValueError('Checkpoint schemas differ; mixed runtime evidence cannot be joined')
    trace = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}
    time = trace['time_s']
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError('Recorded timestamps must be strictly increasing')
    step = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), step, atol=1e-9, rtol=0):
        raise ValueError('Checkpoint gap detected; an uncut playback requires a continuous record')
    return trace, dict(development_prefix=not final.exists(),
                       available_source_time_range_s=[float(time[0]), float(time[-1])],
                       sources=sources, source_sample_interval_s=step)


def _sphere(scene, position, radius, color):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0., 0.]),
                       position, np.eye(3).ravel(), np.asarray(color))
    scene.ngeom += 1


def _target_cue(scene, start, target):
    if scene.ngeom >= scene.maxgeom or np.linalg.norm(target - start) < 1e-7:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.ones(3),
                       np.zeros(3), np.eye(3).ravel(), np.array([.68, .73, .79, .65]))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .00035, start, target)
    scene.ngeom += 1


def control_intent_label(trace, row, finger, path_kinds):
    """Combine recorded phase and sensors; do not call local OFF actuator OFF.

    These remain explanatory intent labels, not proof of the applied actuator
    command. Safe measured contact alone cannot assign an MCC mode in an air or
    unknown phase. Actual joint angles and forces remain separate evidence.
    """
    phase = str(trace['phase'][row])
    if phase in {'LIFT','TRANSFER','APPROACH','PREPARED_TRANSFER','PREPARED_APPROACH'}:
        return 'AIR / REPOSITION'
    if phase == 'FORCE_RELIEF':
        return 'FORCE RELIEF'
    selected = trace.get('coordinated_recovery_finger')
    if selected is not None and int(selected[row]) == finger:
        if phase == 'FINGER_STABILIZE':
            return 'CONTACT STABILIZE'
        if phase == 'FINGER_RECOVERY':
            stage = trace.get('coordinated_recovery_stage')
            sequence = trace.get('coordinated_recovery_sequence')
            kind = path_kinds.get(int(sequence[row])) if sequence is not None else None
            if kind == 'WRIST_PIVOT' and (stage is None or int(stage[row]) == 0):
                return 'PLANNED WRIST MOTION'
            return 'PLANNED FINGER MOTION'
    force = float(trace['forces_n'][row,finger])
    actual = bool(trace['active'][row,finger])
    mcc_phases = {'SLIDE','SELECT','LOCAL_SEARCH','REACQUIRE','NEAR_BRIDGE','FINGER_RECOVERY','FINGER_STABILIZE'}
    if phase in mcc_phases and actual and np.isfinite(force) and .2 <= force < 8.:
        return 'MCC CONTACT HOLD'
    if 'finger_search_state' in trace:
        state = str(trace['finger_search_state'][row, finger])
        if state == 'DISABLED':
            return 'WAITING DURING RECOVERY' if phase in {'FINGER_RECOVERY','FINGER_STABILIZE'} else 'LOCAL SEARCH OFF'
        return state
    return 'SEEK PROPOSAL' if trace['finger_search_active'][row, finger] else 'NO SEEK PROPOSAL'


def render(directory, *, start=0., end=None, speed=1., filename='active_fingers_closeup.mp4',
           close_azimuth=-50., close_elevation=24., overview=False, events_source=None,
           translucent_context_finger=None, display_run_label=None):
    directory = Path(directory).resolve()
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError('Playback speed must be positive')
    if not np.isfinite(close_azimuth) or not np.isfinite(close_elevation) or not -89 <= close_elevation <= 89:
        raise ValueError('Expected finite camera angles and elevation in [-89, 89] degrees')
    if Path(filename).name != filename or not filename.endswith('.mp4'):
        raise ValueError('Output filename must be a local .mp4 basename')
    if translucent_context_finger is not None and translucent_context_finger not in range(4):
        raise ValueError('Translucent context focus must be a measured finger index from 0 to 3')
    if overview and translucent_context_finger is not None:
        raise ValueError('Translucent context is for the right closeup only')
    if display_run_label is not None and (not isinstance(display_run_label, str) or not 1 <= len(display_run_label) <= 32):
        raise ValueError('Display run label must contain 1–32 characters')
    trace, source = load_recorded_trace(directory,through_s=end)
    events_path = Path(events_source).resolve() if events_source is not None else directory/'events.json'
    event_bytes = events_path.read_bytes() if events_path.exists() else None
    events = json.loads(event_bytes.decode('utf-8')) if event_bytes is not None else []
    path_kinds = {int(event['sequence']): event.get('path_kind') for event in events
                  if event.get('kind') == 'COORDINATED_RECOVERY_BEGIN'}
    stop = min(float(trace['time_s'][-1]), end if end is not None else float('inf'))
    start = max(float(trace['time_s'][0]), float(start))
    if start >= stop:
        raise ValueError('Requested playback interval has no completed recorded samples')
    model_bytes = (directory / 'model.xml').read_bytes()
    model = compile_model_xml(model_bytes.decode('utf-8'))
    visual = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'mustard_visual')
    if visual >= 0:
        model.geom_matid[visual] = -1
        model.geom_rgba[visual] = [1., .72, .08, 1.]
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
              for name in (*ARM_JOINT_NAMES, *HAND_JOINT_NAMES)]
    addresses = model.jnt_qposadr[joints]
    base_rgba, base_matid = model.geom_rgba.copy(), model.geom_matid.copy()
    focus_body, focus_geoms = None, set()
    if translucent_context_finger is not None:
        focus_body = int(model.jnt_bodyid[joints[7 + 4 * translucent_context_finger]])
        descendants = {focus_body}
        for body in range(model.nbody):
            if int(model.body_parentid[body]) in descendants:
                descendants.add(body)
        focus_geoms = {geom for geom in range(model.ngeom) if int(model.geom_bodyid[geom]) in descendants}
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'fr3_e05_object')
    center = model.body_pos[object_id] + np.array([0., 0., .095])
    window = (trace['time_s'] >= start) & (trace['time_s'] <= stop)
    focus = trace['tip_positions_world_m'][window].mean(axis=(0, 1))
    first_row = int(np.searchsorted(trace['time_s'], start))
    initial_q = trace['q_rad'][first_row]
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=540, width=720)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    title_font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 20)
    frames = np.r_[np.arange(start, stop, speed / 20.), stop]
    path = directory / filename
    temporary_video = directory / f'{path.stem}.rendering.mp4'
    temporary_frames = []
    targets = trace.get('finger_search_target_positions_world_m', trace.get('finger_search_target_world_m'))
    scope = 'DEVELOPMENT PREFIX' if source['development_prefix'] else 'RECORDED DEVELOPMENT RUN'
    display_scope = (f'{display_run_label} / PREFIX' if source['development_prefix'] else display_run_label) if display_run_label else scope
    with imageio.get_writer(temporary_video, fps=20, codec='libx264', quality=7, macro_block_size=2,
                            output_params=['-threads', '2']) as writer:
        for frame, time in enumerate(frames):
            row = min(int(np.searchsorted(trace['time_s'], time)), len(trace['time_s']) - 1)
            actual_time = float(trace['time_s'][row])
            data.qpos[addresses] = trace['q_rad'][row]
            mujoco.mj_forward(model, data)  # FK playback, never a physical mj_step.
            panels = []
            views = (((130., -30., 'WHOLE OBJECT / SIDE'), (-50., 24., 'WHOLE OBJECT / REVERSE'))
                     if overview else ((130., -30., 'CONTACT / SIDE'),
                                       (close_azimuth, close_elevation, 'FINGER CLOSEUP')))
            for azimuth, elevation, label in views:
                model.geom_rgba[:] = base_rgba
                model.geom_matid[:] = base_matid
                translucent = label == 'FINGER CLOSEUP' and translucent_context_finger is not None
                if translucent:
                    for geom in range(model.ngeom):
                        if geom in focus_geoms:
                            model.geom_matid[geom] = -1
                            model.geom_rgba[geom] = COLORS[translucent_context_finger]
                        elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                            if base_matid[geom] >= 0:
                                model.geom_rgba[geom] = model.mat_rgba[base_matid[geom]]
                            model.geom_matid[geom] = -1
                            model.geom_rgba[geom, 3] = .14
                camera.azimuth, camera.elevation = azimuth, elevation
                camera.distance = .95 if overview else .40 if label == 'FINGER CLOSEUP' else .72
                camera.lookat[:] = (center if overview else trace['tip_positions_world_m'][row].mean(axis=0)
                                   if label == 'FINGER CLOSEUP' else .35 * center + .65 * focus)
                if translucent:
                    camera.lookat[:] = (.5 * trace['tip_positions_world_m'][row, translucent_context_finger]
                                        + .5 * data.xpos[focus_body])
                renderer.update_scene(data, camera=camera, scene_option=options)
                trail_start = int(np.searchsorted(trace['time_s'], max(start, actual_time - 2.)))
                for previous in range(trail_start, row + 1, 20):
                    for finger in np.flatnonzero(trace['active'][previous]):
                        _sphere(renderer.scene, trace['contact_positions_world_m'][previous, finger], .0012, COLORS[finger])
                for finger in range(4):
                    if trace['active'][row, finger]:
                        _sphere(renderer.scene, trace['contact_positions_world_m'][row, finger], .0035, COLORS[finger])
                    proposed = bool(trace.get('finger_search_active', np.zeros((len(trace['time_s']), 4), bool))[row, finger])
                    valid = bool(trace.get('finger_search_target_valid', np.zeros((len(trace['time_s']), 4), bool))[row, finger])
                    if proposed and valid and targets is not None:
                        _target_cue(renderer.scene, trace['tip_positions_world_m'][row, finger], targets[row, finger])
                panel = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, 720, 67), fill=(15, 23, 42))
                display_label = (NAMES[translucent_context_finger] + ' / TRANSLUCENT CONTEXT') if translucent else label
                draw.text((12, 6), f'{display_scope} | {display_label}', font=title_font, fill='white')
                draw.text((12, 34), f'Saved {actual_time:.2f} s | {trace["phase"][row]} | uncut {speed:g}x', font=font, fill='#bce7ff')
                panels.append(panel)
            combined = Image.new('RGB', (1440, 744), (15, 23, 42))
            combined.paste(panels[0], (0, 0))
            combined.paste(panels[1], (720, 0))
            draw = ImageDraw.Draw(combined)
            for x, label in ((12, 'ACTUAL finger joints (deg)'), (470, 'Delta from clip start'),
                             (805, 'FORCE / CONTACT'), (1090, 'CONTROL INTENT')):
                draw.text((x, 548), label, font=font, fill='white')
            for finger, name in enumerate(NAMES):
                q = trace['q_rad'][row, 7 + 4 * finger:11 + 4 * finger]
                delta = q - initial_q[7 + 4 * finger:11 + 4 * finger]
                q_text = ' '.join(f'{angle:7.1f}' for angle in np.rad2deg(q))
                delta_text = ' '.join(f'{angle:+6.1f}' for angle in np.rad2deg(delta))
                force = float(trace['forces_n'][row, finger])
                contact = 'CONTACT' if trace['active'][row, finger] else 'NO CONTACT'
                state = control_intent_label(trace, row, finger, path_kinds)
                y = 579 + 30 * finger
                color = tuple(int(channel * 255) for channel in COLORS[finger][:3])
                draw.text((12, y), name, font=title_font, fill=color)
                draw.text((125, y), q_text, font=small, fill='white')
                draw.text((470, y), delta_text, font=small, fill='#c9d7e8')
                draw.text((805, y), f'{force:5.2f} N / {contact}', font=small, fill=color)
                draw.text((1090, y), state, font=small, fill='#f6ce74' if 'SEEK' in state else '#bcc7d5')
            footer = ('Right: translucent context / colored finger = ACTUAL joints. Dots = measured contacts. Targets and modes are intent.'
                      if translucent_context_finger is not None else
                      'Colored dots = ACTUAL contacts. Gray lines = PLANNED targets. Control mode is intent; measured contact establishes recovery.')
            draw.text((12, 714), footer, font=small, fill='#bce7ff')
            writer.append_data(np.asarray(combined))
            if frame in (0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1):
                frame_path = directory / f'{path.stem}_frame_{frame:04d}.png'
                temporary_frame = frame_path.with_name(frame_path.stem + '.rendering.png')
                combined.save(temporary_frame)
                temporary_frames.append((temporary_frame, frame_path))
            if frame % 200 == 0:
                print(f'ACTIVE_VIDEO {frame}/{len(frames)} saved={actual_time:.2f}s', flush=True)
    from OpenGL import GL
    metadata = dict(source='Read-only recorded native q and sensors; FK playback only; no resimulation or added contact',
                    source_scope=scope, **source, start_s=start, end_s=stop, playback_speed=speed,
                    display_run_label=display_run_label,
                    fps=20, frames=len(frames), encoded_duration_s=len(frames) / 20., uncut=True,
                    displayed_joint_angles='Actual measured q_rad, degrees; not command/bias arrays',
                    search_overlay='Recorded phase plus actual active/force distinguish MCC contact hold, local micro-search, planned full finger/wrist motion, stabilization, waiting and intentional air; labels are intent, actual q/force/contact remain the evidence',
                    mode_overlay_events_sha256=(sha256(event_bytes).hexdigest() if event_bytes is not None else None),
                    mode_overlay_events_source=str(events_path),
                    mode_overlay_path_kinds=path_kinds,
                    model_xml_sha256=sha256(model_bytes).hexdigest(),
                    thin_gray_target_cue='Planned geometry only; never contact or coverage evidence',
                    overview=bool(overview),
                    right_camera=('0.95 m distance; fixed whole-object center' if overview else
                                  '0.40 m distance; look-at follows actual focused fingertip/root midpoint' if translucent_context_finger is not None else
                                  '0.40 m distance; look-at follows actual saved fingertip mean'),
                    right_camera_azimuth_deg=-50. if overview else close_azimuth,
                    right_camera_elevation_deg=24. if overview else close_elevation,
                    translucent_context_finger=translucent_context_finger,
                    translucent_context_scope=('Right panel only: other mesh alpha 0.14, focus finger row color, camera follows actual fingertip/root midpoint; left opaque. Display only; q, force and timing unchanged.'
                                               if translucent_context_finger is not None else None),
                    full_surface_claim=False, renderer=GL.glGetString(GL.GL_RENDERER).decode())
    renderer.close()
    reader = imageio.get_reader(temporary_video)
    try:
        if reader.count_frames() != len(frames) or reader.get_data(len(frames) - 1).shape != (744, 1440, 3):
            raise RuntimeError('Rendered video validation failed; previous video preserved')
    finally:
        reader.close()
    metadata['video_sha256'] = sha256(temporary_video.read_bytes()).hexdigest()
    metadata_path = directory / (path.stem + '_render.json')
    temporary_metadata = metadata_path.with_name(metadata_path.stem + '.rendering.json')
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    # A failed render cannot overwrite an already reviewed, complete video.
    temporary_video.replace(path)
    temporary_metadata.replace(metadata_path)
    for temporary_frame, frame_path in temporary_frames:
        temporary_frame.replace(frame_path)
    print('ACTIVE_VIDEO_DONE', path, flush=True)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--end', type=float)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--filename', default='active_fingers_closeup.mp4')
    parser.add_argument('--close-azimuth', type=float, default=-50.)
    parser.add_argument('--close-elevation', type=float, default=24.)
    parser.add_argument('--events-source', type=Path, help='Optional immutable audited event snapshot for mode overlays')
    parser.add_argument('--translucent-context-finger', type=int, choices=range(4),
                        help='Right display only: show actual selected finger through translucent context; 0 index, 1 middle, 2 ring, 3 thumb')
    parser.add_argument('--display-run-label', help='Short explicit run identity; automatically retains PREFIX on unfinished sources')
    parser.add_argument('--overview', action='store_true', help='Fixed 0.95m whole-object views; all recorded phases retained')
    args = parser.parse_args()
    render(args.directory, start=args.start, end=args.end, speed=args.speed, filename=args.filename,
           close_azimuth=args.close_azimuth, close_elevation=args.close_elevation, overview=args.overview,
           events_source=args.events_source, translucent_context_finger=args.translucent_context_finger,
           display_run_label=args.display_run_label)
