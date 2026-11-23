"""Independent native command replay and measured-state video for I04-c."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np

from .fingertip_hand import DT, GEOMS, SURFACE, FingertipHand
from .fingertip_traversal import PROTOCOL, STATE, Measurements, surface_graph, write_json
from .physics import rotation


def case_surface(case, manifest=None):
    case = Path(case)
    if manifest is None:
        manifest = json.loads((case / 'manifest.json').read_bytes())
    return case / manifest['surface_file'] if 'surface_file' in manifest else SURFACE


def load_case(case):
    case = Path(case)
    manifest = json.loads((case / 'manifest.json').read_bytes())
    blob = (case / 'model.mjb').read_bytes()
    surface = case_surface(case, manifest)
    if (manifest['protocol'] != PROTOCOL or manifest['model_sha256'] != sha256(blob).hexdigest()
            or manifest['surface_sha256'] != sha256(surface.read_bytes()).hexdigest()):
        raise ValueError('I04-c model, protocol or task surface mismatch')
    model = mujoco.MjModel.from_binary_path('case.mjb', {'case.mjb': blob})
    if 'seed_pose_wxyz' not in manifest:
        return FingertipHand(model)
    pose = np.asarray(manifest['seed_pose_wxyz'])
    grasp = pose[:3], rotation(pose[3:]), np.asarray(manifest['seed_joints_rad'])
    hand = FingertipHand(model, object_name=manifest['object_name'], grasp=grasp)
    hand.inward_in_palm[:] = manifest['inward_in_palm']
    return hand


def chunks(case):
    for rec in json.loads((case / 'trace_chunks.json').read_bytes()):
        path = case / rec['file']
        if path.parent != case or sha256(path.read_bytes()).hexdigest() != rec['sha256']:
            raise ValueError(f'Trace integrity check failed: {path.name}')
        with np.load(path, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        if any(len(v) != rec['states'] for v in data.values()):
            raise ValueError(f'Trace row count mismatch: {path.name}')
        yield data


def audit_case(case):
    case = Path(case)
    summary = json.loads((case / 'summary.json').read_bytes())
    h = load_case(case)
    m, d = h.model, mujoco.MjData(h.model)
    with np.load(case / 'initial.npz', allow_pickle=False) as initial:
        mujoco.mj_setState(m, d, initial['integration_state'], STATE)
    mujoco.mj_forward(m, d)
    surface = case_surface(case)
    initial = h.measure(d)
    qualified = initial.eligible.all() if surface == SURFACE else initial.measured_heads.all() and initial.loads.max() < 8.
    if not qualified:
        raise ValueError('Initial grasp is missing qualified fingertip heads')
    stats = Measurements(surface_graph(surface))
    worst = 0.
    preview = []
    observations = []
    excluded = longest_high = 0
    for trace in chunks(case):
        contact_rows = []
        for i, palm in enumerate(trace['palm_command']):
            d.mocap_pos[h.driver], d.mocap_quat[h.driver] = palm[:3], palm[3:]
            d.ctrl[:] = trace['finger_command'][i]
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            err = float(np.max(abs(d.qpos - trace['qpos'][i])))
            if not np.isfinite(err) or err > 1e-9:
                raise ValueError(f'Native state mismatch at step {stats.steps + 1}: {err}')
            worst = max(worst, err)
            touch = h.measure(d)
            if not (np.allclose(touch.forces, trace['normal_force_n'][i], rtol=1e-6, atol=1e-6)
                    and np.allclose(touch.min_fraction, trace['min_head_fraction'][i], rtol=1e-6)
                    and np.allclose(touch.loads, trace['region_load_n'][i], rtol=1e-6, atol=1e-6)
                    and bool(touch.incidental) == trace['incidental_contact'][i]):
                raise ValueError(f'Physical fingertip witness mismatch at step {stats.steps + 1}')
            stats.update(touch)
            excluded += int(touch.loads.max() >= 8.)
            longest_high = max(longest_high, stats.high_steps)
            heads = np.flatnonzero(touch.measured_heads)
            if len(heads):
                rows = np.empty((len(heads), 10), dtype=np.float32)
                rows[:, :4] = np.c_[np.full(len(heads), stats.steps), heads,
                                   touch.forces[heads], np.full(len(heads), touch.loads.max())]
                rows[:, 4:7], rows[:, 7:10] = touch.points[heads], touch.normals[heads]
                contact_rows.append(rows)
            if stats.steps == 1 or stats.steps % 10 == 0 or stats.steps == summary['states']:
                preview.append((stats.steps, d.qpos.copy(), touch.forces.copy(), touch.points.copy(),
                                touch.eligible, trace['phase'][i], stats.coverage.area_fraction))
        if contact_rows:
            observations.append(np.concatenate(contact_rows))
    with np.load(case / 'coverage.npz', allow_pickle=False) as recorded:
        for name, actual in (('visited_vertices', stats.coverage.visited_vertices),
                             ('visited_goals', stats.coverage.visited_goals),
                             ('required_vertices', stats.coverage.graph.required_vertices)):
            np.testing.assert_array_equal(actual, recorded[name], err_msg=name)
    measured = stats.summary()
    for name, value in measured.items():
        if isinstance(value, (float, list)) and value is not None and not (
                isinstance(value, list) and None in value):
            np.testing.assert_allclose(value, summary[name], atol=1e-9, rtol=0., err_msg=name)
        elif value != summary[name]:
            raise ValueError(f'Run/replay summary mismatch: {name}')
    preview_path = case / 'native_preview.npz'
    keys = ('step', 'qpos', 'normal_force_n', 'point_m', 'eligible', 'phase', 'area_fraction')
    np.savez_compressed(preview_path, **{k: np.asarray([row[i] for row in preview])
                                        for i, k in enumerate(keys)})
    contacts_path = case / 'native_head_contacts.npz'
    rows = np.concatenate(observations) if observations else np.empty((0, 10), dtype=np.float32)
    np.savez_compressed(contacts_path, step=rows[:, 0].astype(np.int32),
                        finger=rows[:, 1].astype(np.uint8), normal_force_n=rows[:, 2],
                        max_region_load_n=rows[:, 3], point_m=rows[:, 4:7], normal=rows[:, 7:10])
    report = dict(protocol=PROTOCOL, native_replay_matched=True, max_qpos_error=worst,
                  force_excluded_states=excluded, longest_over_40n_s=longest_high * DT,
                  preview_sha256=sha256(preview_path.read_bytes()).hexdigest(),
                  head_observations=len(rows), head_contacts_sha256=sha256(contacts_path.read_bytes()).hexdigest(),
                  stop_reason=summary['stop_reason'], **measured)
    write_json(case / 'native_audit.json', report)
    return report


def show_split_tips(hand):
    # Draw the actual two-part collision meshes once. Leaving the coincident
    # original STL visible here causes stripes on the illuminated fingertip.
    model = hand.model
    for source, pad in zip(GEOMS, hand.pads):
        original = model.geom(source).id
        connector = model.geom(source + '_connector_collision').id
        model.geom_rgba[connector] = model.geom_rgba[original]
        model.geom_rgba[original, 3] = 0.
        model.geom_rgba[pad] = (.22, .69, .78, 1.)


def recorded_frames(case, states, frames):
    steps = np.rint(np.linspace(1, states, frames)).astype(int)
    traces = list(chunks(case))
    measured = {name: np.concatenate([block[name] for block in traces])[steps - 1]
                for name in ('qpos', 'normal_force_n', 'phase')}
    if sum(len(block['qpos']) for block in traces) != states:
        raise ValueError('Measured-state trace length differs from summary')
    measured['step'] = steps
    measured['point_m'] = np.zeros((len(steps), 4, 3), dtype=np.float32)
    path = case / 'live_head_contacts.npz'
    with np.load(path, allow_pickle=False) as contact:
        order = np.searchsorted(steps, contact['step'])
        present = (order < len(steps)) & (steps[np.minimum(order, len(steps) - 1)] == contact['step'])
        measured['point_m'][order[present], contact['finger'][present]] = contact['point_m'][present]
    return measured, path, sha256(path.read_bytes()).hexdigest()


def render_case(case, output, speed, surface_evidence=None, *, recorded=False, show_moving_contact=False,
                show_low_load=False):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from .pad_contact_visual import camera_from_direction, marker

    case, output = Path(case), Path(output)
    if output.exists() or not np.isfinite(speed) or speed <= 0 or output.suffix.lower() != '.mp4':
        raise ValueError('Choose a new .mp4 output and a positive speed')
    summary = json.loads((case / 'summary.json').read_bytes())
    if not summary['states']:
        raise ValueError('This case contains no executed steps')
    manifest = json.loads((case / 'manifest.json').read_bytes())
    controller = 'Joint planning + MCC'
    if 'policy' in manifest:
        controller = 'Finger Policy + MCC'
    elif manifest.get('branch') == 'fixed-mcc':
        controller = 'Fixed-wrist finger IK + MCC'
    controller = manifest.get('video_label', controller)
    h = load_case(case)
    m, d = h.model, mujoco.MjData(h.model)
    m.vis.global_.offwidth, m.vis.global_.offheight = 640, 400
    m.vis.headlight.ambient[:] = .45
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    camera = mujoco.MjvCamera()
    visual = m.geom('bottle_visual').id
    show_split_tips(h)
    font = ImageFont.load_default(size=19)
    fps = 20 if recorded else 25
    frames = max(1, int(np.ceil(summary['states'] * DT * fps / speed)))
    output.parent.mkdir(parents=True, exist_ok=True)
    if recorded:
        if surface_evidence is None:
            raise ValueError('Recorded replay requires verified contact coverage')
        measured, contact_path, witness_hash = recorded_frames(case, summary['states'], frames)
    else:
        audit, preview = case / 'native_audit.json', case / 'native_preview.npz'
        if not audit.is_file() or not preview.is_file():
            audit_case(case)
        audit_metadata = json.loads(audit.read_bytes())
        if sha256(preview.read_bytes()).hexdigest() != audit_metadata['preview_sha256']:
            raise ValueError('Measured-state preview hash mismatch')
        with np.load(preview, allow_pickle=False) as z:
            measured = {name: z[name] for name in z.files}
        contact_path = case / 'native_head_contacts.npz'
        witness_hash = audit_metadata['head_contacts_sha256']
    steps = measured['step']
    sample = np.arange(frames) if recorded else np.searchsorted(steps, np.linspace(1, steps[-1], frames))
    coverage_label = 'Coverage'
    if surface_evidence is None:
        display_coverage = measured['area_fraction'][sample]
        display_contacts = measured['eligible'][sample]
    else:
        with np.load(surface_evidence, allow_pickle=False) as surface:
            if (surface['head_contacts_sha256'].item() != witness_hash
                    or sha256(contact_path.read_bytes()).hexdigest() != witness_hash):
                raise ValueError('Measured coverage overlay is for a different recording')
            first = surface['all_heads_first_step']
            area = surface['vertex_area_m2']
            unique, index = np.unique(steps[sample], return_inverse=True)
            display_coverage = np.clip((unique[:, None] >= first) @ area / area.sum(), 0., 1.)[index]
        with np.load(contact_path, allow_pickle=False) as contacts:
            at = np.searchsorted(unique, contacts['step'])
            present = (at < len(unique)) & (unique[np.minimum(at, len(unique) - 1)] == contacts['step'])
            display_contacts = np.zeros((len(unique), 4), dtype=bool)
            display_contacts[at[present], contacts['finger'][present]] = True
            display_contacts = display_contacts[index]
        coverage_label = 'Head coverage (25 mm)'
    if show_moving_contact:
        frame_steps = steps[sample] - 1
        with np.load(case / 'moving_contact.npz', allow_pickle=False) as history:
            moving, counts = history['moving'], history['head_count']
            if show_low_load:
                loads = history['region_load_n']
        np.testing.assert_array_equal(counts[frame_steps], display_contacts.sum(axis=1))
        elapsed = np.cumsum(moving)[frame_steps]
        third = ((counts >= 3) & (loads < 8.)) if show_low_load else counts >= 3
        rates = [np.divide(np.cumsum(moving & event)[frame_steps], elapsed,
                           out=np.full(len(sample), np.nan), where=elapsed > 0) for event in (counts >= 2, third)]
        controller = controller.replace(' | Fixed wrist', '')
    center = np.array([0., 0., .095])
    camera_distance = .86
    if 'surface_file' in manifest:
        import trimesh

        mesh = trimesh.load_mesh(case_surface(case, manifest), process=False)
        center = mesh.bounds.mean(axis=0)
        camera_distance = max(.63, 1.6 * float(mesh.extents.max()))
    with mujoco.Renderer(m, height=400, width=640) as renderer, imageio.get_writer(
            output, fps=fps, codec='libx264', quality=7, macro_block_size=2,
            ffmpeg_params=['-preset', 'veryfast'] if recorded else []) as writer:
        for frame, i in enumerate(sample):
            d.qpos[:] = measured['qpos'][i]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            tips = d.geom_xpos[h.pads].mean(axis=0)
            palm = d.xmat[h.palm].reshape(3, 3)
            canvas = Image.new('RGB', (1280, 476 if show_moving_contact else 448), '#16212c')
            for panel in (0, 1):
                look = tips if panel else center
                if panel:
                    direction = palm @ np.array([.30, -1., -.1])
                elif 'surface_file' in manifest:
                    direction = d.qpos[:3] - center + palm @ np.array([.16, -.10, 0.])
                else:
                    direction = np.array([1., -.5, .95])
                camera_from_direction(camera, direction, look, .40 if panel else camera_distance)
                m.geom_rgba[visual, 3] = .25 if panel else 1.
                renderer.update_scene(d, camera=camera, scene_option=options)
                for f in np.flatnonzero(display_contacts[frame]):
                    marker(renderer.scene, measured['point_m'][i, f], .003, (.1, 1., .38, 1.))
                canvas.paste(Image.fromarray(renderer.render()), (640 * panel, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((15, 402), f"{controller}    {steps[i] * DT:7.2f} s    {speed:g}x    "
                      f"{measured['phase'][i]}", font=font, fill='white')
            draw.text((15, 425), f"{coverage_label}  {display_coverage[frame]:.1%}", font=font, fill='white')
            forces = '    '.join(f'{name} {force:.1f}' for name, force in zip(
                ('IF', 'MF', 'RF', 'TH'), measured['normal_force_n'][i]))
            draw.text((654, 412), 'Fingertip force (N)     ' + forces, font=font, fill='white')
            if show_moving_contact:
                active = display_contacts[frame]
                draw.rectangle((0, 448, 1279, 475), fill='#213442')
                draw.text((15, 451), f'Active tips  {active.sum()}/4', font=font, fill='white')
                for f, name in enumerate(('IF', 'MF', 'RF', 'TH')):
                    draw.text((183 + f * 43, 451), name, font=font, fill='#92edbb' if active[f] else '#a6b1bf')
                percentages = [f'{rate[frame]:.2%}' if np.isfinite(rate[frame]) else '—' for rate in rates]
                third_label = '3+ under 8 N' if show_low_load else '3+ tips'
                if show_low_load:
                    draw.text((390, 451), f'Peak region  {loads[frame_steps[frame]]:.1f} N', font=font, fill='white')
                draw.text((654, 451), f'Moving time   2+ tips  {percentages[0]}     {third_label}  {percentages[1]}',
                          font=font, fill='white')
            writer.append_data(np.asarray(canvas))
    return dict(video=str(output), frames=frames, speed=speed, native_states=int(steps[-1]))
