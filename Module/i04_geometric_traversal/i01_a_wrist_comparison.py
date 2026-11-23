"""Compare completed i01-a runs on the same surface guide, in time and by distance."""
import argparse
from pathlib import Path

import numpy as np

from .fingertip_benchmark import digest, read
from .fingertip_replay import chunks
from .fingertip_traversal import DT, write_json
from .i01_a_report import moving_wrist_mask


OBJECTS = ('mustard', 'pear', 'cracker')
BRANCHES = ('fixed-dp', 'fixed-mcc', 'fixed-grasp')


def matched_guide(original, candidate):
    a, b = (read(root / 'plan/wrist_plan.json') for root in (original, candidate))
    for key in ('model_sha256', 'surface_sha256', 'calibration_sha256', 'seed_pose_wxyz', 'seed_joints_rad'):
        if a[key] != b[key]:
            raise ValueError(f'{original.name} has mismatched {key}')
    with np.load(original / 'plan/planned_surface.npz') as old, np.load(candidate / 'plan/planned_surface.npz') as new:
        original_guide = old['guide_m']
        prefix_knots = b['guide_prefix_knots']
        np.testing.assert_array_equal(new['guide_m'][prefix_knots:], original_guide)
        prefix = float(new['arc_m'][prefix_knots])
        length = float(old['arc_m'][-1])
        np.testing.assert_allclose(new['arc_m'][-1] - prefix, length, rtol=0., atol=1.e-8)
    return length, prefix, a, b


def measures(case, length, prefix):
    record = read(case / 'score.json')
    if not record['motion']['route_completed'] or record['reconstruction_status'] != 'written':
        raise ValueError(f'Expected completed scan and reconstruction in {case}')
    with np.load(case / 'moving_contact.npz') as history:
        moving, count, load = (history[key] for key in ('moving', 'head_count', 'region_load_n'))
        progress = history['progress_m']
    shared = np.clip(progress - prefix, 0., length)
    if np.any(np.diff(shared) < -1.e-9) or abs(shared[-1] - length) > 1.e-8:
        raise ValueError(f'Invalid shared guide progress in {case}')
    # Each simulation step receives only the forward distance it covered.
    # Constant-progress ticks and the entry segment contribute zero distance.
    two = count >= 2
    covered = np.cumsum(np.diff(shared, prepend=0.) * two)
    edges = np.linspace(0., length, 17)
    by_section = np.diff(np.interp(edges, shared, covered)) / np.diff(edges)
    time_rate = float(two[moving].mean())
    np.testing.assert_allclose(time_rate, record['motion']['moving_at_least_two_heads'])
    return dict(
        moving_s=record['motion']['moving_s'], two=time_rate, three=record['motion']['moving_at_least_three_heads'],
        four=record['motion']['moving_four_heads'], two_below_8n=float((two & (load < 8.))[moving].mean()),
        three_below_8n=record['motion']['moving_three_heads_below_8n'],
        two_common_guide_time=float(two[moving & (progress >= prefix)].mean()),
        two_common_guide_distance=float(covered[-1] / length), two_by_equal_distance=by_section.tolist(),
        contact_by_finger=record['motion']['moving_each_head'], moving_over_40n=float((load[moving] > 40.).mean()),
        region_load_p95_n=float(np.percentile(load[moving], 95)), peak_load_n=record['peak_load_n'],
        longest_over_40n_s=record['longest_over_40n_s'], full_loss_s=record['motion']['moving_no_head_longest_s'],
        quality=record['quality'], directly_touched_sectors=sum(s['directly_touched_vertices'] > 0
            for s in record['coverage_by_direction'].values()), execution_and_reconstruction_s=record['seconds_to_written'],
    )


def box_edges(original, candidate, prefix):
    """Compare two-tip contact on shared guide segments close to two box faces."""
    import trimesh

    mesh = trimesh.load_mesh(original / 'plan/surface.obj', process=False)
    frame, sizes = trimesh.bounds.oriented_bounds(mesh)
    with np.load(original / 'plan/planned_surface.npz') as planned:
        guide, arc = planned['guide_m'], planned['arc_m']
    guide = trimesh.transform_points(guide, frame)
    edge_band_m = .010
    metrics = dict(edge_band_m=edge_band_m)
    for root, offset, name in ((original, 0., 'original'), (candidate, prefix, 'candidate')):
        with np.load(root / 'fixed-dp_r01/moving_contact.npz') as history:
            station = np.clip(history['progress_m'] - offset, 0., arc[-1])
            positions = np.column_stack([np.interp(station, arc, xyz) for xyz in guide.T])
            distance = np.abs(np.abs(positions) - sizes / 2.)
            edge = (distance <= edge_band_m).sum(axis=1) >= 2
            advances = np.diff(station, prepend=0.)
            two = history['head_count'] >= 2
            low = two & (history['region_load_n'] < 8.)
        regions = {}
        for label, mask in (('edge_band', edge), ('other', ~edge)):
            distance_m = float(advances[mask].sum())
            regions[label] = dict(distance_m=distance_m,
                                  two=float(advances[mask & two].sum() / distance_m) if distance_m else None,
                                  two_below_8n=float(advances[mask & low].sum() / distance_m) if distance_m else None)
        metrics[name] = regions
    return metrics


def compare(baseline, candidate):
    output = {}
    for name in OBJECTS:
        a, b = baseline / name, candidate / name
        length, prefix, old_plan, new_plan = matched_guide(a, b)
        result = dict(shared_guide_m=length, entry_m=prefix,
                      planning_s=[old_plan['geometry_planning_s'], new_plan['geometry_planning_s']],
                      scan_duration_s=[old_plan['nominal_total_s'], new_plan['nominal_total_s']], controls={})
        for branch in BRANCHES:
            old, new = (root / f'{branch}_r01' for root in (a, b))
            first, second = (read(root / 'manifest.json') for root in (old, new))
            if (digest(old / 'initial.npz') != digest(new / 'initial.npz') or first['controller'] != second['controller']
                    or read(old / 'stopwatch.json')['seed'] != read(new / 'stopwatch.json')['seed']):
                raise ValueError(f'{name} {branch} has unmatched initial state, seed or controller')
            if branch == 'fixed-dp':
                for key in ('checkpoint_sha256', 'inference_seed', 'denoise_steps', 'sampler', 'schedule',
                            'control_s', 'action_points', 'waypoint_s', 'observation_history', 'cuda_graph', 'ipc'):
                    if first['policy'][key] != second['policy'][key]:
                        raise ValueError(f'{name} has mismatched policy {key}')
            result['controls'][branch] = dict(original=measures(old, length, 0.), candidate=measures(new, length, prefix))
        for root in (a, b):
            scores = read(root / 'i01_a_results.json')
            if not any(t['repeat'] == 1 and set(t['cases']) == set(BRANCHES) for t in scores['fixed_triplets']):
                raise ValueError(f'Common wrist was not verified in {root}')
        if name == 'cracker':
            result['dp_box_edges'] = box_edges(a, b, prefix)
        output[name] = result
    return output


def plot(root, results):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9.6, 6.6), sharex=True, sharey=True, layout='constrained')
    for ax, name, title in zip(axes, OBJECTS, ('Mustard bottle', 'Pear', 'Cracker box')):
        paired = results[name]['controls']['fixed-dp']
        for key, color, label in (('original', '#797f8a', 'Original wrist'),
                                  ('candidate', '#146d68', 'New wrist')):
            bins = 100. * np.array(paired[key]['two_by_equal_distance'])
            ax.stairs(bins, np.linspace(0, 100, len(bins) + 1), baseline=None, color=color,
                      linewidth=2.2, label=label)
        ax.set(title=title, xlim=(0., 100.), ylim=(0., 102.), yticks=(0, 50, 100))
        ax.grid(axis='y', alpha=.22)
    axes[0].legend(loc='lower right', ncols=2, fontsize=9)
    axes[-1].set_xlabel('Distance along the shared surface guide (%)')
    fig.supylabel('Distance with at least two contacting fingertips (%)', fontsize=11)
    fig.suptitle('i01-a  ·  single_all + MCC', fontweight='bold')
    fig.savefig(root / 'two_finger_contact_by_distance.png', dpi=160)
    plt.close(fig)


def video(baseline, candidate, results):
    from contextlib import ExitStack
    from tempfile import TemporaryDirectory

    import imageio.v2 as imageio
    from imageio_ffmpeg import count_frames_and_secs
    from PIL import Image, ImageDraw, ImageFont

    font, header = ImageFont.load_default(size=18), ImageFont.load_default(size=23)
    with TemporaryDirectory(dir=candidate) as temp:
        output = Path(temp) / 'single_all_old_new_wrist.mp4'
        with imageio.get_writer(output, fps=20, codec='libx264', quality=7, macro_block_size=2,
                                ffmpeg_params=['-preset', 'veryfast']) as writer:
            for name, title in zip(OBJECTS, ('Mustard bottle', 'Pear', 'Cracker box')):
                stations = np.linspace(0., results[name]['shared_guide_m'], 400)
                with ExitStack() as readers:
                    panels = []
                    for root, label, prefix in ((baseline, 'Original wrist', 0.),
                                                (candidate, 'New wrist', results[name]['entry_m'])):
                        obj = root / name
                        source = obj / 'fixed-dp.mp4'
                        frames, _ = count_frames_and_secs(str(source))
                        stream = readers.enter_context(imageio.get_reader(source))
                        with np.load(obj / 'fixed-dp_r01/moving_contact.npz') as h:
                            # Match recorded_frames(), including the first and last simulation steps.
                            ticks = np.rint(np.linspace(1, len(h['time_s']), frames)).astype(int) - 1
                            progress = h['progress_m'][ticks]
                            when = np.clip(np.searchsorted(progress, stations + prefix), 1, len(ticks) - 1)
                            left = abs(progress[when - 1] - stations - prefix) <= abs(progress[when] - stations - prefix)
                            frame = when - left.astype(int)
                            panels.append((stream, label, frame, h['time_s'][ticks[frame]],
                                           h['head_count'][ticks[frame]], h['region_load_n'][ticks[frame]]))
                    for i, station in enumerate(stations):
                        canvas = Image.new('RGB', (1024, 756), '#16232e')
                        draw = ImageDraw.Draw(canvas)
                        draw.text((14, 6), f'i01-a    single_all + MCC    |    {title}', font=header, fill='white')
                        draw.text((685, 9), f'Shared guide progress   {station / stations[-1]:6.1%}',
                                  font=font, fill='white')
                        for row, (stream, label, frame, time, fingers, load) in enumerate(panels):
                            top = 42 + 354 * row
                            draw.text((14, top), f'{label}     {time[i]:.2f} s', font=font, fill='white')
                            color = '#90ebbb' if fingers[i] >= 2 else '#e9bf80'
                            draw.text((347, top), f'Contacting fingertips    {fingers[i]}/4', font=font, fill=color)
                            draw.text((685, top), f'Max regional load    {load[i]:.1f} N', font=font, fill='white')
                            views = stream.get_data(int(frame[i]))[:400]
                            canvas.paste(Image.fromarray(views).resize((1024, 320), Image.Resampling.LANCZOS), (0, top + 27))
                        writer.append_data(np.asarray(canvas))
        output.replace(candidate / output.name)


def joint_reference(root):
    """Score the archived i01-b joint-generated wrist with the i01-a motion thresholds."""
    records = []
    for case in sorted(root.glob('fixed-dp_r[0-9][0-9]')):
        trace = list(chunks(case))
        poses = np.concatenate([part['qpos'][:, :7] for part in trace])
        slide = np.concatenate([part['phase'] == 'slide' for part in trace])
        moving = moving_wrist_mask(poses, slide)
        if not moving.any():
            raise ValueError(f'No moving slide phase in {case}')
        with np.load(case / 'live_head_contacts.npz') as contacts:
            head = np.zeros((len(slide), 4), dtype=bool)
            head[contacts['step'] - 1, contacts['finger']] = True
        records.append(dict(case=case.name, seed=read(case / 'stopwatch.json')['seed'],
                            moving_s=float(moving.sum() * DT), two=float((head[moving].sum(axis=1) >= 2).mean())))
    if not records:
        raise ValueError(f'No fixed-wrist joint-generated DP reference in {root}')
    return dict(source=root.name, object='mustard', phase='slide', records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--joint-reference', type=Path, help='Archived i01-b mustard runs on the joint-generated wrist')
    parser.add_argument('--video', action='store_true', help='Align existing single_all recordings by shared guide distance')
    args = parser.parse_args()
    results = compare(args.baseline, args.candidate)
    summary = dict(baseline=args.baseline.name, candidate=args.candidate.name, objects=results)
    if args.joint_reference is not None:
        summary['joint_reference'] = joint_reference(args.joint_reference)
    write_json(args.candidate / 'wrist_selection_comparison.json', summary)
    plot(args.candidate, results)
    if args.video:
        video(args.baseline, args.candidate, results)
    for name in OBJECTS:
        row = results[name]['controls']['fixed-dp']
        print(f"{name}: {row['original']['two']:.2%} -> {row['candidate']['two']:.2%}", flush=True)


if __name__ == '__main__':
    main()
