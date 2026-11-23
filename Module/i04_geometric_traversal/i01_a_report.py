"""Verify i01-a pairings, score actual moving contact, and render measured results."""
import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import trimesh
from scipy.ndimage import uniform_filter1d

from .fingertip_benchmark import digest, read, read_contacts, score_case
from .fingertip_hand import reference_grasp
from .fingertip_replay import audit_case, case_surface, chunks, render_case
from .fingertip_surface_report import METRIC_SEED, MESH_SAMPLES, render_summary
from .fingertip_traversal import DT, surface_graph, write_json
from .i01_a import BRANCHES, EXPERIMENT, FIXED_BRANCHES
from .i01_a_wrist import wrist_digest


GATE = dict(minimum_moving_three_heads_below_8n=.95, maximum_moving_no_head_s=.10,
            sustained_over_40n_s=.20, route_completed=True)
MOTION_WINDOW_S = .05
MIN_LINEAR_SPEED_M_S = .001
MIN_ANGULAR_SPEED_RAD_S = .02
SHOULDER_ARC_M = (.025, .075)
LABELS = {
    'fixed-grasp': 'Fixed wrist / Fixed grasp (position servo)',
    'fixed-mcc': 'Fixed wrist / MCC force feedback',
    'fixed-dp': 'Fixed wrist / single_all + MCC',
    'adaptive-mcc': 'Tactile wrist feedback / Fixed grasp + MCC',
    'adaptive-dp': 'Tactile wrist feedback / single_all + MCC',
}


def longest(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return float(DT * (edges[1::2] - edges[::2]).max(initial=0))


def moving_wrist_mask(measured, scans):
    steps = np.arange(len(measured))
    half = round(MOTION_WINDOW_S / DT / 2)
    a, b = np.maximum(steps - half, 0), np.minimum(steps + half, len(steps) - 1)
    xyz, quat = measured[:, :3], measured[:, 3:]
    speeds = np.linalg.norm(xyz[b] - xyz[a], axis=1) / (np.maximum(1, b - a) * DT)
    cos = np.einsum('ij,ij->i', quat[a], quat[b])
    angular = 2 * np.arccos(np.clip(abs(cos), 0., 1.)) / (np.maximum(1, b - a) * DT)
    return scans & ((speeds >= MIN_LINEAR_SPEED_M_S) | (angular >= MIN_ANGULAR_SPEED_RAD_S))


def wrist_measurements(case, record, plan):
    trace = list(chunks(case))
    measured = np.concatenate([part['qpos'][:, :7] for part in trace])
    commands = np.concatenate([part['palm_command'] for part in trace])
    scans = np.concatenate([part['phase'] == 'scan' for part in trace])
    loads = np.concatenate([part['region_load_n'].max(axis=1) for part in trace])
    incident = np.concatenate([part['incidental_contact'] for part in trace])
    steps = np.arange(len(measured))
    xyz = measured[:, :3]
    moving = moving_wrist_mask(measured, scans)
    with np.load(case / 'live_head_contacts.npz', allow_pickle=False) as contact:
        head = np.zeros((len(steps), 4), dtype=bool)
        head[contact['step'] - 1, contact['finger']] = True
    count = head.sum(axis=1)
    with np.load(case / 'wrist_progress.npz', allow_pickle=False) as progress:
        arc = progress['route_progress_m']
        cursor = progress['nominal_cursor']
    if len(arc) != len(measured) or np.any(np.diff(cursor) < -1.e-9):
        raise ValueError(f'Invalid route progress in {case.name}')
    with np.load(case / 'nominal_wrist.npz', allow_pickle=False) as nominal:
        tape = nominal['palm_command']
        if wrist_digest(tape) != plan['nominal_wrist_sha256']:
            raise ValueError(f'Invalid nominal pose digest in {case.name}')
        if record['branch'].startswith('fixed'):
            np.testing.assert_array_equal(commands, tape[:len(commands)])
    if record['branch'] == 'fixed-grasp':
        initial_joints = np.asarray(plan['seed_joints_rad']) if 'seed_joints_rad' in plan else reference_grasp()[2]
        for part in trace:
            finger_commands = part['finger_command']
            np.testing.assert_array_equal(finger_commands, np.broadcast_to(initial_joints, finger_commands.shape))
    tracker_error = np.linalg.norm(commands[:, :3] - xyz, axis=1)
    np.savez_compressed(case / 'moving_contact.npz', time_s=(steps + 1) * DT, moving=moving,
                        head_count=count, region_load_n=loads, progress_m=arc)

    def fraction(condition, window=moving):
        return float(np.mean(condition[window])) if window.any() else None

    horizon = min(plan['nominal_wrist_steps'] - 1, len(arc) - 1)
    shoulder = moving & (arc >= SHOULDER_ARC_M[0]) & (arc < SHOULDER_ARC_M[1]) & ('object_name' not in plan)
    return dict(
        moving_s=float(moving.sum() * DT), scanning_hold_s=float((scans & ~moving).sum() * DT),
        moving_at_least_one_head=fraction(count >= 1), moving_at_least_two_heads=fraction(count >= 2),
        moving_at_least_three_heads=fraction(count >= 3),
        moving_two_heads_below_8n=fraction((count >= 2) & (loads < 8.)),
        moving_four_heads=fraction(count == 4), moving_three_heads_below_8n=fraction((count >= 3) & (loads < 8.)),
        moving_each_head=[fraction(column) for column in head.T],
        moving_incidental_contact=fraction(incident), moving_no_head_longest_s=longest(moving & (count == 0)),
        moving_under_three_heads_longest_s=longest(moving & (count < 3)),
        scanning_no_head_longest_s=longest(scans & (count == 0)),
        moving_under_8n=fraction(loads < 8.), moving_peak_region_load_p95_n=(
            float(np.percentile(loads[moving], 95)) if moving.any() else None),
        shoulder_moving_s=float(shoulder.sum() * DT),
        shoulder_three_heads=fraction(count >= 3, shoulder),
        shoulder_four_heads=fraction(count == 4, shoulder),
        shoulder_three_heads_below_8n=fraction((count >= 3) & (loads < 8.), shoulder),
        shoulder_peak_region_load_n=float(loads[shoulder].max()) if shoulder.any() else None,
        max_wrist_tracking_error_mm=float(tracker_error.max() * 1000.),
        wrist_tracking_error_p95_mm=float(np.percentile(tracker_error, 95) * 1000.),
        actual_wrist_distance_m=float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()),
        progress_fraction=float(arc[-1] / plan['path_length_m']),
        progress_at_nominal_deadline_fraction=float(arc[horizon] / plan['path_length_m']),
        route_completed=record['stop_reason'] == 'WRIST_TRAJECTORY_COMPLETE',
    )


def passes(record):
    contact = record['motion']['moving_three_heads_below_8n']
    return bool(contact is not None and contact >= GATE['minimum_moving_three_heads_below_8n'] and
                record['motion']['moving_no_head_longest_s'] <= GATE['maximum_moving_no_head_s'] and
                record['longest_over_40n_s'] < GATE['sustained_over_40n_s'] and
                record['motion']['route_completed'])


def summarize(records):
    fields = ('seconds_to_written', 'executed_s', 'initial_setup_s', 'wrist_generation_s',
              'wrist_feedback_s', 'motion_wall_s', 'reconstruction_s', 'peak_load_n', 'longest_over_40n_s')
    numeric = {key: [row[key] for row in records] for key in fields}
    for section in ('quality', 'motion', 'computing'):
        for key in records[0][section]:
            values = [row[section].get(key) for row in records]
            if all(type(value) in (int, float) for value in values):
                numeric[key] = values
    return dict(trials=len(records), stable_trials=sum(row['passed_moving_contact'] for row in records),
                metrics={key: dict(median=float(np.median(values)), minimum=min(values), maximum=max(values))
                         for key, values in numeric.items()})


def coverage_directions(case, graph):
    """Measure six area-weighted sectors around the object in its local frame."""
    bounds = graph.mesh.bounds
    relative = (graph.vertices_m - bounds.mean(axis=0)) / np.ptp(bounds, axis=0)
    axis = abs(relative).argmax(axis=1)
    signed = relative[np.arange(len(relative)), axis] >= 0.
    with np.load(case / 'measured_area.npz', allow_pickle=False) as area:
        weights, all_heads, below_8n, direct = (area[name] for name in (
            'vertex_area_m2', 'all_heads', 'region_below_8n', 'all_heads_registered'))
    sectors = {}
    for i, coord in enumerate('xyz'):
        for positive, sign in ((False, '-'), (True, '+')):
            region = (axis == i) & (signed == positive)
            denominator = weights[region].sum()
            if denominator > 0.:
                sectors[sign + coord] = dict(area_m2=float(denominator), directly_touched_vertices=int((region & direct).sum()),
                    covered_25mm=float(weights[region & all_heads].sum() / denominator),
                    covered_25mm_below_8n=float(weights[region & below_8n].sum() / denominator))
    return sectors, bool(all_heads.all())


def score(root):
    watches = sorted(root.glob('*/stopwatch.json'))
    if not watches:
        raise ValueError('No completed i01-a trials found')
    graph = surface_graph(case_surface(watches[0].parent))
    truth, _ = trimesh.sample.sample_surface(graph.mesh, MESH_SAMPLES, seed=METRIC_SEED + 1)
    records = []
    for watch in watches:
        case = watch.parent
        manifest = read(case / 'manifest.json')
        plan = read(case / 'wrist_plan.json')
        record = score_case(case, graph, truth, quality_gate=None)
        record['case'] = case.name
        timing = read(case / 'summary.json')['control_timing']
        record['motion'] = wrist_measurements(case, record, plan)
        record['computing'] = dict(mcc_s=timing['mcc'].get('total_s', 0.))
        if record['branch'] == 'fixed-grasp' and (manifest['controller']['mcc_feedback_enabled'] or
                                                   timing['mcc']['calls'] != 0):
            raise ValueError(f'Unexpected force feedback in {case.name}')
        if 'inference_ms' in timing:
            info = timing['inference_ms']
            record['computing'].update(inference_s=info.get('total_s', 0.), dp_calls=info['calls'],
                                       inference_p50_ms=info.get('p50_ms'),
                                       inference_p95_ms=info.get('p95_ms'),
                                       dp_worker_startup_s=manifest['policy']['process_startup_s'],
                                       dp_chunk_rpc_s=timing['chunk_rpc_ms'].get('total_s', 0.),
                                       dp_install_s=timing['chunk_install_ms'].get('total_s', 0.),
                                       dp_observation_ring_s=timing['local_observation_ms'].get('total_s', 0.))
        low_force = record['coverage']['region_below_8n']['area_fraction_by_radius_mm']
        record['quality'].update(coverage_10mm_below_8n=low_force['10'], coverage_25mm_below_8n=low_force['25'])
        if 'object_name' in plan:
            sectors, all_surface = coverage_directions(case, graph)
            record.update(object_name=plan['object_name'], coverage_by_direction=sectors,
                          whole_surface_within_25mm=all_surface)
            record['quality']['minimum_direction_25mm'] = min(x['covered_25mm'] for x in sectors.values())
        record['passed_moving_contact'] = passes(record)
        record['reconstruction_status'] = read(case / 'reconstruction.json')['status']
        if record['experiment'] != EXPERIMENT or manifest['planning']['motion_variables'] != 6:
            raise ValueError(f'Wrist-only experiment signature mismatch in {case.name}')
        write_json(case / 'score.json', record)
        records.append(record)
    pairs, fixed_triplets = [], []
    for repeat in sorted({row['repeat'] for row in records}):
        rows = [r for r in records if r['repeat'] == repeat]
        signatures = set()
        for row in rows:
            case = root / row['case']
            manifest = read(case / 'manifest.json')
            signatures.add((row['seed'], row['nominal_wrist_sha256'], digest(case / 'initial.npz'),
                            manifest['model_sha256'], manifest['surface_sha256'],
                            tuple(manifest['controller']['parameters'].items())))
        if len(signatures) != 1:
            raise ValueError(f'Initial state, wrist reference, or MCC settings differ in repeat {repeat}')
        by_branch = {row['branch']: row for row in rows}
        fixed_rows = [row for row in rows if row['branch'].startswith('fixed')]
        if len({row['wrist_sha256'] for row in fixed_rows}) > 1:
            raise ValueError(f'Executed fixed wrist tapes differ in repeat {repeat}')
        if all(branch in by_branch for branch in FIXED_BRANCHES):
            fixed_triplets.append(dict(repeat=repeat, wrist_sha256=fixed_rows[0]['wrist_sha256'],
                                       cases={branch: by_branch[branch]['case'] for branch in FIXED_BRANCHES}))
        for mode in ('fixed', 'adaptive'):
            a, b = by_branch.get(mode + '-mcc'), by_branch.get(mode + '-dp')
            if a and b:
                pairs.append(dict(mode=mode, repeat=repeat, dp_minus_mcc_s=b['seconds_to_written'] - a['seconds_to_written'],
                                  same_executed_wrist=a['wrist_sha256'] == b['wrist_sha256']))
    groups = {branch: summarize(selected) for branch in BRANCHES
              if (selected := [row for row in records if row['branch'] == branch])}
    plan = read(root / records[0]['case'] / 'wrist_plan.json')
    result = dict(experiment=EXPERIMENT, plan=plan, moving_contact_gate=GATE,
                  motion_measurement=dict(window_s=MOTION_WINDOW_S, linear_m_s=MIN_LINEAR_SPEED_M_S,
                                          angular_rad_s=MIN_ANGULAR_SPEED_RAD_S, shoulder_arc_m=SHOULDER_ARC_M),
                  groups=groups, pairs=pairs, fixed_triplets=fixed_triplets, trials=records)
    write_json(root / 'i01_a_results.json', result)
    return result, graph, truth


def figures(root, result, graph, truth):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from .fingertip_r_gpis import voxel_representatives

    first = {row['branch']: row for row in result['trials'] if row['repeat'] == 1}
    peak_coverage = 100 * max(row['quality']['coverage_25mm'] for row in first.values())
    coverage_limit = min(105., 5. * (np.ceil(peak_coverage / 5.) + 1.))
    comparisons = (
        ('fixed', ('fixed-mcc', 'fixed-dp'), ('MCC force feedback', 'single_all + MCC')),
        ('adaptive', ('adaptive-mcc', 'adaptive-dp'), ('MCC force feedback', 'single_all + MCC')),
        ('fixed_three_way', ('fixed-mcc', 'fixed-dp', 'fixed-grasp'),
         ('MCC force feedback', 'single_all + MCC', 'Fixed grasp')),
    )
    if any('object_name' in row for row in first.values()):
        comparisons = comparisons[-1:]
    for mode, case_names, case_labels in comparisons:
        if not all(name in first for name in case_names):
            continue
        rows = [first[name] for name in case_names]
        bounds = max(row['executed_s'] for row in rows)
        time = np.r_[np.arange(0., bounds, .1), bounds]
        branches, meshes = [], []
        for row in rows:
            case = root / row['case']
            points, _, _, _ = read_contacts(case)
            sample_steps = np.minimum(np.rint(time / DT), row['states'])
            with np.load(case / 'measured_area.npz', allow_pickle=False) as area:
                weights = area['vertex_area_m2']
                curve = {}
                for field in ('all_heads', 'region_below_8n'):
                    covered = sample_steps[:, None] >= area[f'{field}_first_step']
                    curve[field] = dict(area_fraction_over_time_25mm=covered @ weights / weights.sum())
            branches.append(dict(coverage=curve, voxels=points[voxel_representatives(points)],
                                 reconstruction=dict(accuracy=row['r_gpis_accuracy'])))
            mesh_file = case / 'r_gpis.ply'
            meshes.append(trimesh.load_mesh(mesh_file, process=False) if mesh_file.exists() else None)
        directory = root / (mode + '_figures')
        directory.mkdir(exist_ok=True)
        render_summary(directory, time, branches, graph, meshes, truth,
                       case_labels=case_labels,
                       coverage_limit=coverage_limit, legend_location='upper left')

    colors = {'fixed-grasp': '#466193', 'fixed-mcc': '#168278', 'fixed-dp': '#a64919',
              'adaptive-mcc': '#168278', 'adaptive-dp': '#a64919'}
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10.2, 7.6), layout='constrained')
    for branch in BRANCHES:
        if branch not in first:
            continue
        with np.load(root / first[branch]['case'] / 'moving_contact.npz', allow_pickle=False) as series:
            t = series['time_s'][::25]
            quorum = ((series['head_count'] >= 3) & (series['region_load_n'] < 8.)).astype(float)
            style = dict(label=LABELS[branch], color=colors[branch],
                         linestyle='--' if branch.startswith('adaptive') else '-', linewidth=1.8)
            axes[0].plot(t, 100 * series['progress_m'][::25] / result['plan']['path_length_m'], **style)
            axes[1].plot(t, 100 * uniform_filter1d(quorum, size=250, mode='nearest')[::25], **style)
            axes[2].plot(t, series['region_load_n'][::25], **style)
    for axis, label in zip(axes, ('Route progress (%)', '≥3 heads under 8 N (%)', 'Peak region load (N)')):
        axis.set_ylabel(label)
        axis.grid(alpha=.23)
    axes[0].legend(fontsize=8.6, loc='upper left', ncols=2)
    axes[1].set_ylim(-3, 103)
    axes[2].set_yscale('symlog', linthresh=2.)
    axes[2].axhline(8., color='#5c6670', linewidth=.8, linestyle=':')
    axes[2].set_xlabel('Simulated time (s)')
    fig.suptitle('i01-a | Measured contact and wrist progress · repeat 1')
    fig.savefig(root / 'contact_and_progress.png', dpi=165)
    plt.close(fig)
    if all(branch in first and 'coverage_by_direction' in first[branch] for branch in FIXED_BRANCHES):
        fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.8), layout='constrained')
        sectors = ('-x', '+x', '-y', '+y', '-z', '+z')
        fields = (('covered_25mm', 'All fingertip contacts'),
                  ('covered_25mm_below_8n', 'Contacts below 8 N in every region'))
        for ax, (field, title) in zip(axes, fields):
            data = np.array([[first[branch]['coverage_by_direction'][side][field] for side in sectors]
                              for branch in FIXED_BRANCHES])
            image = ax.imshow(data * 100., vmin=0., vmax=100., cmap='viridis', aspect='auto')
            for row, col in np.ndindex(data.shape):
                ax.text(col, row, f'{data[row, col]:.1%}', va='center', ha='center',
                        color='black' if data[row, col] > .68 else 'white', fontsize=10)
            ax.set(xticks=np.arange(len(sectors)), xticklabels=sectors, yticks=np.arange(len(FIXED_BRANCHES)),
                   yticklabels=('Fixed grasp', 'MCC force feedback', 'single_all + MCC'), title=title)
        fig.suptitle(f'{first[FIXED_BRANCHES[0]]["object_name"]} · 25 mm measured coverage')
        axes[-1].set_xlabel('Sector in the object frame')
        fig.colorbar(image, ax=axes, label='Surface area covered (%)')
        fig.savefig(root / 'coverage_by_direction.png', dpi=165)
        plt.close(fig)


def videos(root, result, speed):
    for record in result['trials']:
        if record['repeat'] != 1:
            continue
        case = root / record['case']
        output = root / (record['branch'] + '.mp4')
        if output.exists():
            continue
        if 'object_name' in record:
            print(f'Render measured states from {case.name}', flush=True)
            with TemporaryDirectory(dir=root) as temp:
                video = Path(temp) / output.name
                render_case(case, video, speed, surface_evidence=case / 'measured_area.npz', recorded=True,
                            show_moving_contact=True)
                video.replace(output)
            continue
        print(f'Native replay and render {case.name}', flush=True)
        audit = audit_case(case)
        with np.load(case / 'live_head_contacts.npz', allow_pickle=False) as live, np.load(
                case / 'native_head_contacts.npz', allow_pickle=False) as replay:
            for key in ('step', 'finger', 'point_m', 'normal', 'normal_force_n', 'max_region_load_n'):
                np.testing.assert_allclose(live[key], replay[key], atol=1.e-6, rtol=0.)
        with np.load(case / 'measured_area.npz', allow_pickle=False) as area:
            overlay = {key: area[key] for key in area if key != 'head_contacts_sha256'}
        overlay_path = case / 'video_area.npz'
        np.savez_compressed(overlay_path, **overlay, head_contacts_sha256=audit['head_contacts_sha256'])
        with TemporaryDirectory(dir=root) as temp:
            video = Path(temp) / output.name
            render_case(case, video, speed, surface_evidence=overlay_path)
            video.replace(output)
    comparisons = (('fixed_comparison.mp4', ('fixed-mcc', 'fixed-dp')),
                   ('fixed_three_way.mp4', ('fixed-mcc', 'fixed-dp', 'fixed-grasp')))
    if any('object_name' in record for record in result['trials']):
        comparisons = comparisons[-1:]
    for name, group in comparisons:
        fixed = [root / f'{branch}.mp4' for branch in group]
        comparison = root / name
        if comparison.exists() or not all(path.exists() for path in fixed):
            continue
        from imageio_ffmpeg import get_ffmpeg_exe

        with TemporaryDirectory(dir=root) as temp:
            video = Path(temp) / comparison.name
            inputs = [item for path in fixed for item in ('-i', str(path))]
            filters = ''.join(f'[{i}:v]' for i in range(len(group)))
            subprocess.run([get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                            *inputs, '-filter_complex', f'{filters}vstack=inputs={len(group)}:shortest=1',
                            '-c:v', 'libx264', '-crf', '20', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart', str(video)], check=True)
            video.replace(comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--media', action='store_true')
    parser.add_argument('--speed', type=float, default=2.)
    args = parser.parse_args()
    objects = [args.root / name for name in ('mustard', 'pear', 'cracker')
               if (args.root / name / 'plan' / 'wrist_plan.json').is_file()]
    results = []
    for directory in objects or [args.root]:
        if objects and not all((directory / f'{branch}_r01' / 'stopwatch.json').is_file()
                               for branch in FIXED_BRANCHES):
            raise ValueError(f'Complete all three full-surface controls in {directory} before aggregating')
        result, graph, truth = score(directory)
        figures(directory, result, graph, truth)
        results.append((directory, result))
        print(f'i01-a report: {len(result["trials"])} trials in {directory}', flush=True)
    if len(objects) == 3:
        write_json(args.root / 'i01_a_full_results.json', dict(experiment=EXPERIMENT,
            objects={directory.name: result for directory, result in results}))
    if args.media:
        for directory, result in results:
            videos(directory, result, args.speed)


if __name__ == '__main__':
    main()
