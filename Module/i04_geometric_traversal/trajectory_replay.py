"""Interactive MuJoCo playback of the published I04 A/B measured trajectories.

Reads saved native states and contact metrics; never calls mj_step, runs a
planner, or writes to an experiment directory. Rendering may skip samples.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import OrderedDict
from hashlib import sha256
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

# `python -m Module...` imports Module/__init__.py (and NumPy) before this
# module. Delegate to the standalone file if that happened with uncapped BLAS
# pools; setting environment variables after NumPy import is too late. Direct
# script invocation avoids even the small bootstrap process's NumPy import.
# No shared/frozen package source needs to change for this viewer-only limit.
_THREAD_ENV = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
if __name__ == '__main__':
    _restart = 'numpy' in sys.modules and any(os.environ.get(k) != '1' for k in _THREAD_ENV)
    for _key in _THREAD_ENV:
        os.environ[_key] = '1'
    if _restart:
        import subprocess
        print('Starting replay with one numerical-library thread...', flush=True)
        raise SystemExit(subprocess.call(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=os.environ.copy()))

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    'A': ROOT / 'Module/generated/i04a_frozen_b01_20261021',
    'B': ROOT / 'Module/generated/i04b_palm_recovery02_20261027',
    'B-SAFE': ROOT / 'Module/generated/i04b_palm_recovery02_20261027',
}
KEYS = ('time_s', 'q_rad', 'qvel_rad_s', 'sensor_command_rad',
        'qacc_warmstart', 'phase', 'coverage', 'forces_n', 'safe_belly',
        'belly_contact_positions_world_m')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_context(case):
    """Load verified original geometry without importing planners or plots.

    Only the in-memory compiler/rendering settings change. The 256 collision
    pieces, visible meshes, joint tree and recorded dynamics settings remain.
    VFS names support the original Windows Unicode workspace.
    """
    case = Path(case).resolve()
    catalog = read(case / 'portable_assets.json')
    xml = (case / 'model.xml').read_bytes()
    if sha256(xml).hexdigest() != catalog['model_sha256']:
        raise ValueError('Original model.xml hash mismatch')
    mapping = {}
    shared = read(ROOT / 'Module/evidence/whole_hand_release_assets_20261019.json')
    for base, entries in ((ROOT, shared['assets']), (case, catalog['assets'])):
        for entry in entries:
            path = (base / entry['path']).resolve()
            if not path.is_relative_to(base):
                raise ValueError(f'Asset path escapes its directory: {path}')
            mapping[entry['original_file']] = (path, entry['sha256'])
    root = ET.fromstring(xml)
    root.find('compiler').set('usethread', 'false')
    visual = root.find('visual')
    visual.find('global').set('offwidth', '640')
    visual.find('global').set('offheight', '480')
    quality = visual.find('quality')
    quality.set('shadowsize', '1024')
    quality.set('offsamples', '0')
    assets = {}
    for index, element in enumerate(root.findall('./asset/*[@file]')):
        path, expected = mapping[element.get('file')]
        content = path.read_bytes()
        if sha256(content).hexdigest() != expected:
            raise ValueError(f'Incomplete or changed asset: {path}; try git lfs pull')
        name = f'vfs_{index}{path.suffix}'
        assets[name] = content
        element.set('file', name)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets=assets)
    if (model.nq, model.nv, model.nu) != (23, 23, 23):
        raise ValueError('Expected the recorded FR3 + LEAP 23-joint model')

    def ids(kind, names):
        values = np.array([mujoco.mj_name2id(model, kind, name) for name in names])
        if np.any(values < 0):
            raise ValueError(f'Missing original model names: {names}')
        return values

    joints = ids(mujoco.mjtObj.mjOBJ_JOINT,
                 [f'fr3v2_joint{i}' for i in range(1, 8)] + [str(i) for i in range(16)])
    actuators = ids(mujoco.mjtObj.mjOBJ_ACTUATOR,
                    [f'fr3_arm_act_{i}' for i in range(1, 8)] + [f'act_{i}' for i in range(16)])
    tips = ids(mujoco.mjtObj.mjOBJ_GEOM, [f'fr3_tip_{i}' for i in range(1, 5)])
    return SimpleNamespace(model=model, data=mujoco.MjData(model),
                           qpos=model.jnt_qposadr[joints], dof=model.jnt_dofadr[joints],
                           actuator_ids=actuators, handles=SimpleNamespace(tip_geom_ids=tips))


def marker(scene, point):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.002, 0., 0.]),
                       np.asarray(point), np.eye(3).ravel(), np.array([1., .1, .55, 1.]))
    scene.ngeom += 1


class RecordedTrajectory:
    """One-chunk cache, independent of the total trajectory duration."""

    def __init__(self, case):
        self.case = Path(case).resolve()
        self.chunks = read(self.case / 'summary.json')['trace_chunks']
        if not self.chunks:
            raise ValueError('No completed trajectory chunks')
        self.starts = [float(c['start_s']) for c in self.chunks]
        if any(b <= a for a, b in zip(self.starts, self.starts[1:])):
            raise ValueError('Chunk start times must increase')
        self.cache = OrderedDict()
        self.verified = set()
        self.start = float(self.load(0)['time_s'][0])
        self.end = float(self.load(len(self.chunks) - 1)['time_s'][-1])

    def load(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        chunk = self.chunks[index]
        # Release the old chunk before decompressing a new one.
        self.cache.clear()
        path = (self.case / chunk['file']).resolve()
        if not path.is_relative_to(self.case):
            raise ValueError('Trace path escapes the case directory')
        if index not in self.verified:
            if digest(path) != chunk['sha256']:
                raise ValueError(f'Incomplete or changed trace: {path}; try git lfs pull')
            self.verified.add(index)
        with np.load(path, allow_pickle=False) as archive:
            count = len(archive['time_s']) - int(chunk.get('skip_last', 0))
            values = {key: archive[key][:count] for key in KEYS}
            for key in ('palm_clearance_m', 'nearest_palm_geom'):
                if key in archive.files:
                    values[key] = archive[key][:count]
        if count < 1 or np.any(np.diff(values['time_s']) <= 0):
            raise ValueError(f'Invalid recorded timestamps in {path}')
        self.cache[index] = values
        return values

    def at(self, timestamp):
        timestamp = float(np.clip(timestamp, self.start, self.end))
        chunk_index = max(0, bisect_right(self.starts, timestamp) - 1)
        values = self.load(chunk_index)
        index = max(0, int(np.searchsorted(values['time_s'], timestamp, side='right')) - 1)
        return {key: value[index] for key, value in values.items()}


def apply_state(context, row):
    """Restore a measured frame; any interactive perturbation is overwritten."""
    data = context.data
    data.qpos[context.qpos] = row['q_rad']
    data.qvel[context.dof] = row['qvel_rad_s']
    data.ctrl[context.actuator_ids] = row['sensor_command_rad']
    data.qacc_warmstart[:] = row['qacc_warmstart']
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    data.time = float(row['time_s'])
    mujoco.mj_forward(context.model, data)


def camera_default(camera):
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [.3, 0., .49]
    camera.distance = 1.05
    camera.azimuth = 130.
    camera.elevation = -25.


def play(context, trajectory, args, start, end):
    import mujoco.viewer

    keys = SimpleQueue()
    model, data = context.model, context.data
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'mustard_visual')
    if object_geom >= 0:
        model.geom_matid[object_geom] = -1
        model.geom_rgba[object_geom] = [.96, .72, .08, 1.]
    for geom in range(model.ngeom):
        if 'normal_marker' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ''):
            model.geom_rgba[geom, 3] = 0.
    model.vis.headlight.ambient[:] = [.25, .25, .25]
    model.vis.headlight.diffuse[:] = [.45, .45, .45]
    timestamp, speed, paused = start, args.speed, args.paused
    transparent = False
    apply_state(context, trajectory.at(timestamp))
    print('Space: pause | Left/Right: seek 10s | Up/Down: speed x2 / x0.5', flush=True)
    print('R: restart | T: transparent object | F: focus hand | C: reset camera', flush=True)
    print('Mouse: rotate/pan/zoom. Close the viewer to exit. End frame stays visible.', flush=True)
    with mujoco.viewer.launch_passive(
        model, data, key_callback=keys.put,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            camera_default(viewer.cam)
            viewer.opt.geomgroup[3:] = 0
            viewer.opt.sitegroup[:] = 0
        last = time.perf_counter()
        next_text_update = 0.
        while viewer.is_running():
            tick = time.perf_counter()
            if not paused:
                timestamp = min(end, timestamp + (tick - last) * speed)
            last = tick
            focus = reset_camera = False
            appearance_changed = False
            while True:
                try:
                    key = keys.get_nowait()
                except Empty:
                    break
                if key == 32:  # Space.
                    paused = not paused
                elif key == 262:  # GLFW right.
                    timestamp = min(end, timestamp + 10.)
                elif key == 263:
                    timestamp = max(start, timestamp - 10.)
                elif key == 265:
                    speed = min(128., speed * 2.)
                elif key == 264:
                    speed = max(.125, speed / 2.)
                elif key == ord('R'):
                    timestamp = start
                elif key == ord('T'):
                    transparent = not transparent
                    appearance_changed = True
                elif key == ord('F'):
                    focus = True
                elif key == ord('C'):
                    reset_camera = True
            if timestamp >= end:
                paused = True
            row = trajectory.at(timestamp)
            with viewer.lock():
                apply_state(context, row)
                if object_geom >= 0:
                    model.geom_rgba[object_geom, 3] = .15 if transparent else 1.
                if reset_camera:
                    camera_default(viewer.cam)
                if focus:
                    viewer.cam.lookat[:] = data.geom_xpos[context.handles.tip_geom_ids].mean(axis=0)
                    viewer.cam.distance = .4
                viewer.user_scn.ngeom = 0
                for finger in np.flatnonzero(row['safe_belly']):
                    marker(viewer.user_scn, row['belly_contact_positions_world_m'][finger])
            if tick >= next_text_update and hasattr(viewer, 'set_texts'):
                next_text_update = tick + .1
                palm_text = ''
                if 'palm_clearance_m' in row:
                    distance = float(row['palm_clearance_m'])
                    value = '>=6.00' if distance >= .006-1e-12 else f'{1000*distance:.2f}'
                    palm_text = f'\nRecorded palm clearance: {value} mm'
                viewer.set_texts((None, None,
                    f'I04 {args.case} | RECORDED STATE REPLAY\n'
                    f'{float(row["time_s"]):.3f} / {end:.3f} s | {speed:g}x | '
                    f'{"PAUSED" if paused else "PLAYING"}\n'
                    f'{row["phase"]}\n'
                    f'Recorded coverage: {100 * float(row["coverage"]):.2f}% | '
                    f'Qualified pads: {int(row["safe_belly"].sum())}/4\n'
                    'Recorded force [N]: ' + ' / '.join(f'{f:.2f}' for f in row['forces_n']) + palm_text + '\n'
                    'Space pause | arrows seek/speed | R restart | T transparent | F hand | C camera',
                    None))
            # Copy only integration state on ordinary frames; full sync is
            # needed when T changes model.geom_rgba in the local viewer.
            viewer.sync(state_only=not appearance_changed)
            time.sleep(max(0., 1. / args.fps - (time.perf_counter() - tick)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, help='A, B (alias B-safe), or a completed case directory')
    parser.add_argument('--speed', type=float, default=1., help='Playback speed, default 1x')
    parser.add_argument('--start', type=float, default=0., help='Start at simulation seconds')
    parser.add_argument('--end', type=float, help='Stop at simulation seconds; hold final frame')
    parser.add_argument('--paused', action='store_true', help='Open paused')
    parser.add_argument('--fps', type=float, default=30., help='Maximum display rate (default 30)')
    parser.add_argument('--check', action='store_true', help='Validate sampled frames without a GUI')
    args = parser.parse_args()
    if not np.isfinite(args.speed) or not .125 <= args.speed <= 128:
        parser.error('--speed must be between 0.125 and 128')
    if not np.isfinite(args.fps) or not 1 <= args.fps <= 240:
        parser.error('--fps must be between 1 and 240')
    case = CASES.get(args.case.upper(), Path(args.case)).resolve()
    if case == CASES['A']:
        print('A is the retained partial result with its original collision model; '
              'it is not a palm-repaired full-coverage result.', flush=True)
    if not (case / 'portable_assets.json').is_file():
        parser.error(f'Published portable_assets.json is missing in {case}')
    if (not (case / 'summary.json').is_file()
            or (args.case.upper() == 'B-SAFE' and not (case / 'run_complete.json').is_file())):
        parser.error('This case has no completed trajectory summary yet; wait for the experiment to finish')
    print(f'Loading recorded trajectory: {case.name}', flush=True)
    trajectory = RecordedTrajectory(case)
    print('Loading verified model (single compiler thread, low-memory rendering)...', flush=True)
    context = load_context(case)
    start, end = max(trajectory.start, args.start), args.end
    end = trajectory.end if end is None else min(end, trajectory.end)
    if (not np.isfinite(args.start) or (args.end is not None and not np.isfinite(args.end))
            or args.start < trajectory.start or not start < end):
        parser.error(f'Choose a nonempty interval within {trajectory.start:.3f}..{trajectory.end:.3f} seconds')
    print(f'{case.name}: {len(trajectory.chunks)} chunks; {start:.3f}..{end:.3f} s', flush=True)
    if args.check:
        samples = sorted(set([start, end, (start + end) / 2] +
                             [t for t in trajectory.starts[1:2] if start <= t <= end]))
        for timestamp in samples:
            row = trajectory.at(timestamp)
            apply_state(context, row)
            np.testing.assert_array_equal(context.data.qpos[context.qpos], row['q_rad'])
            assert np.all(np.isfinite(context.data.geom_xpos))
            if 'palm_clearance_m' in row:
                m = context.model
                palm_ids = set(map(int, np.flatnonzero(m.geom_contype == 8)))
                if len(palm_ids) != 6:
                    raise ValueError('Palm-clearance trace requires all six enabled palm collision parts')
                obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'fr3_e05_object')
                objects = set(map(int, np.flatnonzero((m.geom_bodyid == obj) & (m.geom_contype != 0))))
                clearance = .006
                for ct in context.data.contact:
                    a, b = int(ct.geom1), int(ct.geom2)
                    if (a in palm_ids and b in objects) or (b in palm_ids and a in objects):
                        clearance = min(clearance, float(ct.dist))
                np.testing.assert_allclose(clearance, row['palm_clearance_m'], atol=1e-9, rtol=0)
        print(json.dumps(dict(status='PASSED_REPLAY_STATE_LOAD_CHECK', samples=samples,
                              checked_chunks=len(trajectory.verified), native_physics_steps=0,
                              viewer_window_tested=False, case_files_written=0)), flush=True)
    else:
        play(context, trajectory, args, start, end)


if __name__ == '__main__':
    main()
