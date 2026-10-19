"""Independent native replay audit and an uncut selected-finger probe video."""
from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import mujoco
import numpy as np

from Module.e05_physics.scene import FINGERS
from Module.fr3_leap import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from Module.fr3_leap.model import compile_model_xml
from .physics import contact_measurement


def load(directory):
    directory = Path(directory)
    with np.load(directory / 'trace.npz') as archive:
        trace = {key: archive[key] for key in archive.files}
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    model = compile_model_xml((directory / 'model.xml').read_text(encoding='utf-8'))
    def ids(kind, names):
        values = np.array([mujoco.mj_name2id(model, kind, name) for name in names])
        assert np.all(values >= 0), names
        return values
    joint = ids(mujoco.mjtObj.mjOBJ_JOINT, (*ARM_JOINT_NAMES, *HAND_JOINT_NAMES))
    actuator = ids(mujoco.mjtObj.mjOBJ_ACTUATOR,
                   (*[f'fr3_arm_act_{i}' for i in range(1, 8)], *[f'act_{i}' for i in range(16)]))
    sites = ids(mujoco.mjtObj.mjOBJ_SITE, [f.pad_site.replace('e05_', 'fr3_') for f in FINGERS])
    geoms = ids(mujoco.mjtObj.mjOBJ_GEOM, [f.proxy_geom.replace('e05_', 'fr3_') for f in FINGERS])
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'fr3_e05_object')
    objects = np.flatnonzero(model.geom_bodyid == body)
    handles = SimpleNamespace(model=model, tip_geom_ids=geoms, object_geom_ids=objects,
                              object_geom_id=int(objects[0]))
    return directory, trace, manifest, summary, handles, joint, actuator, sites


def audit(directory):
    directory, tr, manifest, summary, h, joints, actuators, sites = load(directory)
    assert manifest['protocol'] in ('i04-single-finger-withdraw-recover.probe.v1',
                                     'i04-single-finger-withdraw-recover.paired.v2')
    assert not manifest['whole_traversal_evidence'] and manifest['coverage_authority'] == 'NONE'
    assert not summary['coverage_counted'] and not summary['whole_traversal_evidence']
    digest = sha256((directory / 'trace.npz').read_bytes()).hexdigest()
    assert digest == summary['trace_sha256']
    for name, expected in manifest['source_hashes'].items():
        relative = PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert sha256((directory / 'source_snapshot' / relative).read_bytes()).hexdigest() == expected
    runtime = directory / 'source_snapshot/Module/i04_geometric_traversal/finger_recovery_probe.py'
    tree = ast.parse(runtime.read_text(encoding='utf-8'))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    for node in ast.walk(run):
        if isinstance(node, ast.Call):
            assert not (isinstance(node.func, ast.Name) and node.func.id == 'set_joint_state')
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
        assert not any(isinstance(item, ast.Attribute) and item.attr == 'qpos'
                       for target in targets for item in ast.walk(target))
    model, data = h.model, mujoco.MjData(h.model)
    qpos, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    np.testing.assert_allclose(np.diff(tr['time_s']), model.opt.timestep, atol=1e-12, rtol=0)
    with np.load(directory / 'initial.npz') as initial:
        np.testing.assert_array_equal(initial['qpos'][qpos], tr['q_rad'][0])
        np.testing.assert_array_equal(initial['qvel'][dofs], tr['qvel_rad_s'][0])
        np.testing.assert_array_equal(initial['ctrl'][actuators], tr['sensor_command_rad'][0])
        data.qpos[:] = initial['qpos']
        data.qvel[:] = initial['qvel']
        data.ctrl[:] = initial['ctrl']
    data.qacc_warmstart[:] = tr['qacc_warmstart'][0]
    mujoco.mj_forward(model, data)
    maxima = dict(q_rad=0., qvel_rad_s=0., force_n=0., contact_position_m=0., normal=0., tip_fk_m=0.)
    active_errors = 0
    finger = int(manifest['selected_finger'])
    centers = []
    for row in range(len(tr['time_s'])):
        # No state restoration after the initial row: every next state must
        # arise from one native integration step using the recorded command.
        maxima['q_rad'] = max(maxima['q_rad'], float(np.max(abs(data.qpos[qpos] - tr['q_rad'][row]))))
        maxima['qvel_rad_s'] = max(maxima['qvel_rad_s'], float(np.max(abs(data.qvel[dofs] - tr['qvel_rad_s'][row]))))
        maxima['tip_fk_m'] = max(maxima['tip_fk_m'], float(np.max(abs(data.site_xpos[sites] - tr['tip_positions_world_m'][row]))))
        forces, points, normals, active, unsafe, _ = contact_measurement(h, data, .2)
        maxima['force_n'] = max(maxima['force_n'], float(np.max(abs(forces - tr['forces_n'][row]))))
        maxima['contact_position_m'] = max(maxima['contact_position_m'], float(np.max(abs(points - tr['contact_positions_world_m'][row]))))
        maxima['normal'] = max(maxima['normal'], float(np.max(abs(normals - tr['normals_world'][row]))))
        active_errors += int(np.count_nonzero(active != tr['active'][row]))
        assert not unsafe, (row, unsafe)
        centers.append(data.geom_xpos[h.tip_geom_ids[finger]].copy())
        if row + 1 < len(tr['time_s']):
            data.ctrl[actuators] = tr['command_rad'][row]
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
    assert maxima['q_rad'] < 1e-8 and maxima['qvel_rad_s'] < 1e-7, maxima
    assert maxima['force_n'] < 1e-5 and maxima['contact_position_m'] < 1e-7 and maxima['normal'] < 1e-7, maxima
    assert active_errors == 0
    assert np.max(abs(tr['command_rad'][:, :7] - tr['command_rad'][0, :7])) < 1e-12
    assert np.max(abs(np.diff(tr['command_rad'], axis=0)) / np.diff(tr['time_s'])[:, None]) <= 1.2 + 1e-8
    assert not np.any(tr['finger_search_active'][:, np.arange(4) != finger])
    assert not np.any(tr['finger_search_active'] & tr['active'])
    events = json.loads((directory / 'events.json').read_text(encoding='utf-8'))
    withdraw = next(e for e in events if e['kind'] == 'CONTROLLED_WITHDRAW_BEGIN')
    recovery = next(e for e in events if e['kind'] in ('SELECTED_FINGER_SEARCH_BEGIN', 'RECOVERY_WINDOW_BEGIN'))
    centers = np.array(centers)
    a, b = [int(np.argmin(abs(tr['time_s'] - e['time_s']))) for e in (withdraw, recovery)]
    displacement = float((centers[b] - centers[a]) @ np.array(withdraw['measured_normal']))
    assert .002 <= displacement <= .004
    assert not tr['active'][b, finger]
    safe = ((tr['time_s'] >= recovery['time_s']) & tr['active'][:, finger]
            & (tr['forces_n'][:, finger] >= .2) & (tr['forces_n'][:, finger] <= 8.))
    edges = np.diff(np.r_[False, safe, False].astype(int))
    stable_runs = [(int(a), int(b)-1) for a,b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
                   if tr['time_s'][b-1] - tr['time_s'][a] >= .15 - 1e-10]
    assert bool(stable_runs) == summary['success']
    confirmation_delay = (float(tr['time_s'][stable_runs[0][0]]+.15-recovery['time_s']) if stable_runs else None)
    if manifest['protocol'].endswith('paired.v2'):
        assert summary['stop_reason'] == 'RECOVERY_WINDOW_COMPLETE'
        assert abs(tr['time_s'][-1] - recovery['time_s'] - manifest['recovery']['observation_window_s']) < 1e-10
        if not manifest['recovery']['active_search_enabled']:
            assert not tr['finger_search_active'].any()
            assert not tr['finger_search_enabled'].any()
        if summary['success']:
            assert abs(confirmation_delay - summary['search_to_stable_confirmation_s']) < 1e-10
    result = dict(status='PASSED', scope='Controlled single-finger recovery only; no whole-surface coverage',
        source_snapshots_verified=len(manifest['source_hashes']), source_direct_joint_teleports=0,
        replayed_native_steps=len(tr['time_s'])-1, sensor_reconstructed_states=len(tr['time_s']),
        replay_initializations=1, replay_state_restores_after_initialization=0,
        actual_contact_mismatches=active_errors, maximum_native_replay_errors=maxima,
        selected_finger=finger, independently_measured_withdrawal_m=displacement,
        same_finger_stable_recontact_observed=bool(stable_runs), required_safe_confirmation_s=.15,
        recovery_start_to_confirmation_s=confirmation_delay,
        fixed_recovery_observation_window_s=manifest['recovery'].get('observation_window_s'),
        wrist_command_fixed=True, only_selected_finger_search_applied=True,
        coverage_counted=False, trace_sha256=digest)
    (directory / 'probe_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('PROBE_AUDIT', json.dumps(result), flush=True)
    return result


def audit_pair(directory):
    directory = Path(directory)
    results = {case: audit(directory / case) for case in ('ON', 'OFF')}
    manifests = {case: json.loads((directory / case / 'manifest.json').read_text(encoding='utf-8')) for case in results}
    assert manifests['ON']['source_hashes'] == manifests['OFF']['source_hashes']
    assert manifests['ON']['recovery']['active_search_enabled']
    assert not manifests['OFF']['recovery']['active_search_enabled']
    for value in manifests.values():
        value['recovery'].pop('active_search_enabled')
    assert manifests['ON'] == manifests['OFF'], 'More than the search-layer switch changed'
    with np.load(directory / 'ON/initial.npz') as first, np.load(directory / 'OFF/initial.npz') as second:
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key])
    traces = {}
    for case in results:
        with np.load(directory / case / 'trace.npz') as values:
            traces[case] = dict(values)
    summaries = {case: json.loads((directory / case / 'summary.json').read_text(encoding='utf-8')) for case in results}
    start = summaries['ON']['recovery_started_s']
    assert start == summaries['OFF']['recovery_started_s']
    common_keys = ('time_s', 'q_rad', 'qvel_rad_s', 'command_rad', 'sensor_command_rad',
                   'palm_pose_world', 'tip_positions_world_m', 'contact_positions_world_m',
                   'normals_world', 'forces_n', 'active', 'commanded_withdrawal_m', 'measured_withdrawal_m')
    prefix = traces['ON']['time_s'] < start
    for key in common_keys:
        np.testing.assert_array_equal(traces['ON'][key][prefix], traces['OFF'][key][prefix], err_msg=key)
    result = dict(status='PASSED', same_source_hashes=True, identical_fresh_acquisition=True,
        sole_changed_parameter='recovery.active_search_enabled', identical_pre_recovery_samples=int(prefix.sum()),
        identical_pre_recovery_fields=list(common_keys), recovery_started_s=start,
        observation_window_s=summaries['ON']['recovery_observation_window_s'],
        ON=dict(observed=summaries['ON']['observed_outcome'], first_recontact_delay_s=summaries['ON']['first_recontact_delay_s'],
                confirmation_delay_s=summaries['ON']['search_to_stable_confirmation_s']),
        OFF=dict(observed=summaries['OFF']['observed_outcome'], first_recontact_delay_s=summaries['OFF']['first_recontact_delay_s'],
                 confirmation_delay_s=summaries['OFF']['search_to_stable_confirmation_s']),
        evidence_scope='One controlled selected-finger recovery pair; not whole-surface success/failure',
        cases=results)
    (directory / 'paired_probe_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('PAIRED_PROBE_AUDIT', json.dumps(result), flush=True)
    return result


def render(directory):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from OpenGL import GL
    directory, tr, manifest, summary, h, joints, _, sites = load(directory)
    model = h.model
    visual = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'mustard_visual')
    model.geom_matid[visual] = -1
    model.geom_rgba[visual] = [1., .72, .08, 1.]
    finger = int(manifest['selected_finger'])
    columns = np.asarray([int(name) for name in FINGERS[finger].joint_names]) + 7
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=540, width=720)
    camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    options = mujoco.MjvOption(); options.geomgroup[3:] = 0
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    center = tr['tip_positions_world_m'][:, finger].mean(axis=0)
    frames = np.r_[np.arange(0., tr['time_s'][-1], 1/30.), tr['time_s'][-1]]
    output = directory / 'index_recovery_1x_uncut.mp4'
    with imageio.get_writer(output, fps=30, codec='libx264', quality=8, macro_block_size=2,
                           output_params=['-threads', '2']) as writer:
        for frame, t in enumerate(frames):
            row = min(int(np.searchsorted(tr['time_s'], t)), len(tr['time_s']) - 1)
            data.qpos[model.jnt_qposadr[joints]] = tr['q_rad'][row]
            mujoco.mj_forward(model, data)
            panels = []
            for azimuth, elevation, distance, label in ((135., -25., .40, 'INDEX CLOSE VIEW'),
                                                         (-45., 20., .60, 'REVERSE VIEW')):
                camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
                camera.lookat[:] = center
                renderer.update_scene(data, camera=camera, scene_option=options)
                for f in np.flatnonzero(tr['active'][row]):
                    slot = renderer.scene.ngeom
                    color = np.array([.05, .65, 1., 1.]) if f == finger else np.array([.1, .8, .3, 1.])
                    mujoco.mjv_initGeom(renderer.scene.geoms[slot], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([.0035, 0., 0.]), tr['contact_positions_world_m'][row, f],
                        np.eye(3).ravel(), color)
                    renderer.scene.ngeom += 1
                panel = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, 720, 115), fill=(15, 23, 42))
                contact = bool(tr['active'][row, finger])
                draw.text((10, 5), f'{label} | {t:.3f} s | 1x UNCUT PHYSICAL RECORD', font=font, fill='white')
                draw.text((10, 31), f'{tr["command_phase"][row]} | INDEX {"CONTACT" if contact else "NO CONTACT"} | '
                          f'{tr["forces_n"][row, finger]:.3f} N', font=font, fill='#80ddff' if contact else '#ffca80')
                angles = np.rad2deg(tr['q_rad'][row, columns])
                draw.text((10, 57), 'Actual index joints (deg): ' + ', '.join(f'{angle:+.2f}' for angle in angles), font=font, fill='white')
                draw.text((10, 85), f'Active index search: {bool(tr["finger_search_active"][row, finger])}  | '
                          'Blue = actual index contact; green = other actual contacts', font=small, fill='#b7cadd')
                panels.append(np.asarray(panel))
            combined = np.concatenate(panels, axis=1)
            writer.append_data(combined)
            if frame in (0, len(frames)//2, len(frames)-1):
                Image.fromarray(combined).save(directory / f'probe_frame_{frame:04d}.png')
    metadata = dict(source='Uncut FK playback of saved native q and actual sensors; no resimulation in video',
        speed=1., fps=30, frames=len(frames), simulation_start_s=0., simulation_end_s=float(tr['time_s'][-1]),
        controlled_perturbation=True, whole_traversal_evidence=False, coverage_counted=False,
        renderer=GL.glGetString(GL.GL_RENDERER).decode(), selected_finger=finger,
        displayed_angles='Actual measured joint angles, degrees; not references',
        trace_sha256=sha256((directory / 'trace.npz').read_bytes()).hexdigest())
    (directory / 'index_recovery_1x_uncut_render.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    renderer.close()
    print('PROBE_VIDEO', output.resolve(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--paired', action='store_true')
    args = parser.parse_args()
    render(args.directory) if args.render else audit_pair(args.directory) if args.paired else audit(args.directory)
