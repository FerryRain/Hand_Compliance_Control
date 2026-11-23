"""Geodesic coverage, reconstruction, efficiency and measured-pose video for acitive-02."""

from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import shutil
from time import perf_counter

import mujoco
import numpy as np
import trimesh
from scipy.sparse.csgraph import connected_components

from .experiment import ARMS, CONTROL_DT, digest, read_json, write_json
from .report import COLORS, DISPLAY_NAMES, evaluation_reference, reconstruct, registered_vertices, vertex_coverage
from ..i04_geometric_traversal.contracts import ExperimentConfig
from ..i04_geometric_traversal.oracle import MeasuredCoverage

COVERAGE_TARGETS = (50, 70, 90, 95, 99, 100)
FSCORE_TARGETS = (40, 60, 80, 90)
FULL_TOL = 1.e-10
MULTIFINGER_PROFILES = ('cooperative-surface', 'tactile-gpis-surface')


def coverage_residuals(ledger, covered):
    """Area and locations of distinct uncovered regions on the evaluation mesh."""
    graph = ledger.graph
    remaining = np.flatnonzero(~covered)
    if not len(remaining):
        return dict(uncovered_area_cm2=0., uncovered_regions=0, largest_uncovered_regions=[])
    _, region = connected_components(graph.adjacency[remaining][:, remaining], directed=False)
    weights = ledger.vertex_area[remaining]
    area = np.bincount(region, weights=weights)
    order = np.argsort(-area)[:6]
    patches = []
    for label in order:
        keep = region == label
        center = np.average(graph.vertices_m[remaining[keep]], weights=weights[keep], axis=0)
        patches.append(dict(area_cm2=float(area[label] * 10000), center_mm=(center * 1000).tolist()))
    return dict(uncovered_area_cm2=float(weights.sum() * 10000), uncovered_regions=len(area),
                largest_uncovered_regions=patches)


def evaluate_arm(job):
    case, only_coverage, maximum_load, prefix = job
    run = read_json(case / 'run.json')
    truth, ledger = evaluation_reference()
    started = perf_counter()
    trace_hash = digest(case / 'trace.npz')
    scoring_source = {name: digest(Path(__file__).with_name(name)) for name in ('full_report.py', 'report.py')}
    score_path = case / 'scores.json'
    if not only_coverage and score_path.is_file() and (case / 'coverage.npz').is_file():
        prior = read_json(score_path)
        if (prior.get('trace_sha256') == trace_hash and prior.get('scoring_source_sha256') == scoring_source
                and prior.get('qualified_surface_region_load_n') == maximum_load
                and all(e['mesh'] is None or (case / e['mesh']).is_file() for e in prior['snapshots'])):
            return prior
    with np.load(case / 'trace.npz', allow_pickle=False) as trace:
        load = trace['region_load_n'].max(axis=1)
        qualified = trace['pad_valid'] & (load <= maximum_load)[:, None]
        quorum = qualified.sum(axis=1)
        ticks, fingers = np.nonzero(qualified)
        points = trace['points_m'][ticks, fingers]
        normals = trace['normals_outward'][ticks, fingers]
        raw_contacts = int(trace['pad_valid'].sum())
        total_ticks = len(trace['qpos']) - 1
        wall = trace['wall_s']
        offset = trace['wrist_command'][1:, :3] - trace['wrist_nominal'][1:, :3]
        np.testing.assert_allclose(offset, trace['wrist_offset'][1:, :3], rtol=0., atol=1.e-9)
        difference = np.max(abs(trace['finger_nominal'] - trace['aude_joint_nominal']))
        if (difference == 0.) != (run['arm'] == ARMS[0]):
            raise ValueError(f'Finger command authority differed from the protocol in {case}')
    mapped, good = registered_vertices(ledger, points, normals)
    registered_points = mapped[good]
    registered_ticks = ticks[good]
    by_tick, registered, first_covered = vertex_coverage(ledger, registered_points, registered_ticks, total_ticks,
                                                         return_witness=True)
    wide_ledger = MeasuredCoverage(ledger.graph, ExperimentConfig(coverage_radius_m=.025))
    wide_curve, _, wide_witness = vertex_coverage(wide_ledger, registered_points, registered_ticks, total_ticks,
                                                 return_witness=True)
    covered_area = float(ledger.vertex_area.sum())
    covered = first_covered <= total_ticks
    milestone = {}
    for threshold in COVERAGE_TARGETS:
        if threshold == 100:
            first = int(first_covered.max()) if covered.all() else None
        else:
            hit = np.flatnonzero(by_tick >= threshold / 100 - FULL_TOL)
            first = int(hit[0]) if len(hit) else None
        milestone[str(threshold)] = (dict(sim_s=float(first * CONTROL_DT), wall_s=float(wall[first]))
                                     if first is not None else None)
    wide_full = np.flatnonzero(wide_curve >= 1. - FULL_TOL)
    coverage_files = dict(coverage_10_mm=by_tick, coverage_25_mm=wide_curve, first_covered_tick_10_mm=first_covered)
    cooperative = {}
    if run['profile'] in MULTIFINGER_PROFILES:
        for minimum in (2, 3):
            subset = quorum[registered_ticks] >= minimum
            key = f'coverage_from_at_least_{minimum}_contacts'
            narrow, _ = vertex_coverage(ledger, registered_points[subset], registered_ticks[subset], total_ticks)
            wide, _ = vertex_coverage(wide_ledger, registered_points[subset], registered_ticks[subset], total_ticks)
            coverage_files[f'{key}_10_mm'] = narrow
            cooperative[f'{key}_10_mm'] = float(narrow[-1])
            cooperative[f'{key}_25_mm'] = float(wide[-1])
        cooperative['full_surface_25_mm'] = bool((wide_witness <= total_ticks).all())
        cooperative['majority_multi_contact'] = (run['qualified_at_least_two_contacts'] > .50
                                                and run['qualified_moving_at_least_two_contacts'] > .50)
    np.savez_compressed(case / 'coverage.npz', **coverage_files)
    basic = dict(**run, trace_sha256=trace_hash, scoring_source_sha256=scoring_source,
                 qualified_surface_region_load_n=maximum_load, raw_contacts=raw_contacts, contacts=int(len(ticks)),
                 registered_contacts=registered, coverage_10_mm=float(by_tick[-1]),
                 coverage_25_mm=float(wide_curve[-1]), complete_measured_surface=bool(covered.all()),
                 time_to_100_coverage_25_mm_sim_s=float(wide_full[0] * CONTROL_DT) if len(wide_full) else None,
                 coverage_milestones_10_mm=milestone, mesh_area_m2=covered_area,
                 measured_area_cm2=float(by_tick[-1] * covered_area * 10000),
                 average_coverage_per_sim_min_pp=float((by_tick[-1] - by_tick[0]) * 100 * 60 / run['seconds']),
                 **coverage_residuals(ledger, covered), **cooperative)
    if only_coverage:
        return basic
    if run['profile'] == 'tactile-gpis-surface':
        samples = [round(t / CONTROL_DT) for t in (0, 30, 180, 360, 600, 900) if t < run['seconds']]
        samples.append(total_ticks)
        series = [dict(time_s=at * CONTROL_DT, acquisition_wall_s=float(wall[at]),
                       coverage_10_mm=float(by_tick[at]), mesh=None) for at in samples]
        result = dict(**basic, snapshots=series)
        write_json(score_path, result)
        return result

    reusable = {}
    if prefix is not None and (prefix / 'scores.json').is_file():
        earlier = read_json(prefix / 'scores.json')
        source = earlier.get('scoring_source_sha256', {})
        if (earlier['trace_sha256'] == run['gap_refinement']['prefix_trace_sha256'] == digest(prefix / 'trace.npz')
                and source.get('report.py') == scoring_source['report.py']
                and earlier.get('qualified_surface_region_load_n') == maximum_load):
            reusable = {round(e['time_s'] / CONTROL_DT): e for e in earlier['snapshots']
                        if e['mesh'] is None or (prefix / e['mesh']).is_file()}
    snapshots = sorted({0, total_ticks,
                        *[int(t / CONTROL_DT) for t in (30, 90, 180, 360, 600, 900, 1200, 1500, 1800, 2100)
                          if t < run['seconds']],
                        *[min(total_ticks, int(round(m['sim_s'] / CONTROL_DT))) for t, m in milestone.items()
                          if t in ('70', '90', '95', '100') and m is not None]})
    # Very close milestones share the next saved snapshot. Timing for coverage
    # crossings stays exact to the physical control tick.
    samples = [0]
    for at in snapshots[1:-1]:
        if (at - samples[-1]) * CONTROL_DT >= 20 and (total_ticks - at) * CONTROL_DT >= 12:
            samples.append(at)
    samples.append(total_ticks)
    series = {}
    # Write the complete mesh early so it is available alongside the longer
    # set of intermediate reconstructions.
    for at in (total_ticks, *samples[:-1]):
        if at != total_ticks and at in reusable:
            surface = dict(reusable[at], reconstruction_reused_from_prefix=True)
            if surface['mesh']:
                shutil.copyfile(prefix / surface['mesh'], case / surface['mesh'])
        else:
            keep = ticks <= at
            surface = reconstruct(case, at, points[keep], normals[keep], truth)
        series[at] = dict(surface, time_s=at * CONTROL_DT, acquisition_wall_s=float(wall[at]),
                          coverage_10_mm=float(by_tick[at]))
        print(f'{run["arm"]}: scored {at * CONTROL_DT:.1f}s, F@5 {surface["fscore_5_mm"]:.1%}', flush=True)
    snapshots = [series[at] for at in samples]
    final = snapshots[-1]
    mesh_s = final['mesh_written_wall_s']
    quality_milestones = {str(threshold): next((entry['time_s'] for entry in snapshots
                                               if entry['fscore_5_mm'] >= threshold / 100), None)
                          for threshold in FSCORE_TARGETS}
    result = dict(**basic, reconstructed=final['mesh'], symmetric_mean_mm=final['symmetric_mean_mm'],
                  fscore_5_mm=final['fscore_5_mm'], precision_5_mm=final['precision_5_mm'],
                  recall_5_mm=final['recall_5_mm'], first_scored_snapshot_above_fscore_5_mm_s=quality_milestones,
                  final_reconstruction_wall_s=mesh_s,
                  acquisition_plus_final_reconstruction_wall_s=(run['wall_execution_s'] + mesh_s
                                                                if mesh_s is not None else None),
                  evaluated_wall_s=perf_counter() - started, snapshots=snapshots)
    write_json(score_path, result)
    return result


def compare_initial_plan(root, results):
    if len(results) != 2:
        return
    if results[0]['initial_state_sha256'] != results[1]['initial_state_sha256']:
        raise ValueError('The two runs do not share the same initial integration state')
    plans = []
    for arm in ARMS:
        with (root / arm / 'plans.jsonl').open(encoding='utf-8') as trace:
            plans.append(json.loads(trace.readline()))
    for key in ('wrist_target', 'aude_joint_target', 'virtual_contacts_m'):
        np.testing.assert_allclose(plans[0][key], plans[1][key], atol=1.e-7, rtol=0.)


def score_full(root, only_coverage=False):
    manifest = read_json(root / 'protocol.json')
    if digest(root / 'model.mjb') != manifest['model_sha256'] or digest(root / 'initial.npz') != manifest['initial_sha256']:
        raise ValueError('The stored model or shared starting state changed')
    arms = [name for name in ARMS if (root / name / 'run.json').is_file()]
    prefix = root.parent / manifest['refinement']['prefix'] if 'refinement' in manifest else None
    jobs = [(root / arm, only_coverage, manifest['qualified_surface_region_load_n'], prefix / arm if prefix else None)
            for arm in arms]
    if len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=len(jobs)) as workers:
            results = list(workers.map(evaluate_arm, jobs))
    else:
        results = [evaluate_arm(job) for job in jobs]
    if not results:
        raise RuntimeError('There are no finished execution records in the requested folder')
    compare_initial_plan(root, results)
    for r in results:
        print(r['arm'], f'{r["seconds"]:.1f}s; C10 {r["coverage_10_mm"]:.2%}; C25 {r["coverage_25_mm"]:.2%}',
              flush=True)
    artifact = 'coverage-summary.json' if only_coverage else 'results.json'
    summary = dict(experiment=manifest['experiment'], profile=manifest['profile'], completed_arms=len(results),
                   complete_measured_surface=len(results) == 2 and all(r['complete_measured_surface'] for r in results),
                   runs=results)
    if manifest['profile'] in MULTIFINGER_PROFILES:
        summary['paired_majority_multi_contact'] = len(results) == 2 and all(r['majority_multi_contact'] for r in results)
        summary['paired_full_surface_25_mm'] = len(results) == 2 and all(r['full_surface_25_mm'] for r in results)
    write_json(root / artifact, summary)
    if not only_coverage:
        draw_comparison(root, results)
        draw_coverage_map(root, results)
        if manifest['profile'] in MULTIFINGER_PROFILES:
            draw_contact_distribution(root, results)


def draw_comparison(root, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    if results[0]['profile'] == 'tactile-gpis-surface':
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), facecolor='#f4f7f5', layout='constrained')
        fig.suptitle('active-02-a / Measured tactile coverage', fontsize=17, fontweight='bold')
        for result in results:
            arm = ARMS.index(result['arm'])
            with np.load(root / result['arm'] / 'coverage.npz', allow_pickle=False) as tape:
                for ax, radius in zip(axes, (10, 25)):
                    values = tape[f'coverage_{radius}_mm']
                    times = np.arange(len(values))[::25] * CONTROL_DT
                    ax.plot(times, values[::25], color=COLORS[arm], label=DISPLAY_NAMES[arm], lw=2)
        for ax, radius in zip(axes, (10, 25)):
            ax.set(title=f'{radius} mm geodesic neighborhood', xlabel='Exploration time (simulated seconds)', ylim=(0, 1.03))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis='y', alpha=.22)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].legend(loc='lower right', frameon=False)
        fig.savefig(root / 'comparison.png', dpi=170)
        plt.close(fig)
        return

    truth, _ = evaluation_reference()
    local_end = read_json(root / 'protocol.json').get('local_s', 0.)
    fig = plt.figure(figsize=(12, 7.8), facecolor='#f4f7f5', layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1, .97])
    fig.suptitle(f'{results[0]["experiment"]} / Pear tactile exploration', fontsize=18, fontweight='bold', x=.015, ha='left')
    cov_ax, f_ax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    for r in results:
        arm = ARMS.index(r['arm'])
        with np.load(root / r['arm'] / 'coverage.npz', allow_pickle=False) as tape:
            cov = tape['coverage_10_mm']
        times = np.arange(len(cov))[::25] * CONTROL_DT
        cov_ax.plot(times, cov[::25], color=COLORS[arm], label=DISPLAY_NAMES[arm], lw=2)
        entries = r['snapshots']
        f_ax.plot([x['time_s'] for x in entries], [x['fscore_5_mm'] for x in entries], 'o-', color=COLORS[arm],
                  label=DISPLAY_NAMES[arm], lw=2, markersize=3.8)
    for ax, title in ((cov_ax, 'Physical contact coverage · 10 mm'), (f_ax, 'R-GPIS reconstruction · F-score at 5 mm')):
        ax.set(title=title, xlabel='Exploration time (simulated seconds)', ylim=(0, 1.03))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.axvspan(0, local_end, color='#8ea7a2', alpha=.15)
        ax.grid(axis='y', alpha=.22)
        ax.spines[['top', 'right']].set_visible(False)
    cov_ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    limits = np.vstack((truth.min(axis=0), truth.max(axis=0))) + np.array([[-.024], [.024]])
    for r in results:
        arm = ARMS.index(r['arm'])
        ax = fig.add_subplot(grid[1, arm], projection='3d', facecolor='#f4f7f5')
        ax.scatter(*truth[::12].T, s=.9, color='#80948c', alpha=.16, depthshade=False)
        if p := r['reconstructed']:
            mesh = trimesh.load(root / r['arm'] / p, force='mesh', process=False)
            points, _ = trimesh.sample.sample_surface(mesh, 5000, seed=20261120)
            ax.scatter(*points.T, s=1.6, color=COLORS[arm], alpha=.80, depthshade=False)
        ax.set_xlim(*limits[:, 0])
        ax.set_ylim(*limits[:, 1])
        ax.set_zlim(*limits[:, 2])
        ax.set_box_aspect(np.ptp(limits, axis=0))
        ax.view_init(elev=26, azim=60)
        ax.set_axis_off()
        ax.set_title(f'{DISPLAY_NAMES[arm]}   |   C10 {r["coverage_10_mm"]:.1%}   |   F@5 {r["fscore_5_mm"]:.1%}',
                     fontsize=10)
    fig.savefig(root / 'comparison.png', dpi=170)
    plt.close(fig)


def draw_coverage_map(root, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    _, ledger = evaluation_reference()
    mesh = ledger.graph.mesh
    fig, panels = plt.subplots(2, 2, figsize=(9.4, 7.4), facecolor='#f4f7f5', layout='constrained',
                               subplot_kw={'projection': '3d', 'facecolor': '#f4f7f5'})
    fig.suptitle('Pear surface reached by measured fingertips', fontsize=16, fontweight='bold')
    bounds = mesh.bounds + np.array([[-.007], [.007]])
    light = np.maximum(0., mesh.face_normals @ np.array([.35, .43, .83]))
    for r in results:
        arm = ARMS.index(r['arm'])
        with np.load(root / r['arm'] / 'coverage.npz', allow_pickle=False) as cov:
            covered = cov['first_covered_tick_10_mm'] < len(cov['coverage_10_mm'])
        fraction = covered[mesh.faces].mean(axis=1, keepdims=True)
        color = fraction * to_rgb('#176e77') + (1. - fraction) * to_rgb('#e33e58')
        color *= (.77 + .23 * light)[:, None]
        for side, (elev, azim) in enumerate(((32, 46), (-32, 226))):
            ax = panels[side, arm]
            patch = Poly3DCollection(mesh.triangles, facecolors=color, edgecolors='none')
            ax.add_collection3d(patch)
            ax.set_xlim(*bounds[:, 0])
            ax.set_ylim(*bounds[:, 1])
            ax.set_zlim(*bounds[:, 2])
            ax.set_box_aspect(np.ptp(bounds, axis=0))
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
            title = f'{DISPLAY_NAMES[arm]}   |   {r["coverage_10_mm"]:.2%}' if side == 0 else 'Opposite side'
            ax.set_title(title, fontsize=11, pad=-7)
    fig.legend(handles=[Patch(color='#176e77', label='Within 10 mm of a qualified touch'),
                        Patch(color='#e33e58', label='Beyond 10 mm')], ncol=2,
               loc='outside lower center', frameon=False, fontsize=9)
    plt.savefig(root / 'surface-coverage.png', dpi=165)
    plt.close(fig)


def draw_contact_distribution(root, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    from scipy.ndimage import uniform_filter1d

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), facecolor='#f4f7f5', layout='constrained')
    fig.suptitle(f'{results[0]["experiment"]} / Simultaneous fingertip contact', fontweight='bold', fontsize=17)
    hist, cov_ax, sustained_ax, rhythm_ax = axes.ravel()
    for run in results:
        arm = ARMS.index(run['arm'])
        color, label = COLORS[arm], DISPLAY_NAMES[arm]
        case = root / run['arm']
        distribution = run['qualified_contact_count_fractions']
        bars = hist.bar(np.arange(5) + (arm - .5) * .35, distribution, width=.34, color=color, label=label)
        hist.bar_label(bars, labels=[f'{v:.1%}' if v >= .02 else '' for v in distribution], padding=2, fontsize=8)
        with np.load(case / 'coverage.npz', allow_pickle=False) as measured:
            for label_key, style in (('coverage_10_mm', '-'), ('coverage_from_at_least_2_contacts_10_mm', '--')):
                area = measured[label_key]
                cov_ax.plot(np.arange(len(area))[::25] * CONTROL_DT, area[::25], style, color=color, lw=1.8)
        with np.load(case / 'trace.npz', allow_pickle=False) as trace:
            load = trace['region_load_n'][1:].max(axis=1)
            fingers = trace['pad_valid'][1:].sum(axis=1) * (load <= run['qualified_surface_region_load_n'])
        time = CONTROL_DT * np.arange(1, len(fingers) + 1)
        rhythm_ax.plot(time[::25], uniform_filter1d((fingers >= 3).astype(float), 500)[::25], color=color, lw=1.2)
        total = [run['qualified_at_least_two_contacts'], run['qualified_moving_at_least_two_contacts'],
                 run['qualified_at_least_three_contacts']]
        blocks = sustained_ax.bar(np.arange(3) + (arm - .5) * .35, total, width=.34, color=color)
        sustained_ax.bar_label(blocks, labels=[f'{v:.1%}' for v in total], padding=3, fontsize=9)

    hist.set(title='Simultaneous fingertip count', xlabel='Measured fingers in contact', xticks=range(5), ylim=(0, 1.02))
    hist.legend(loc='upper left', frameon=False, fontsize=8)
    cov_ax.set(title='10 mm measured coverage', xlabel='Simulated seconds', ylim=(0, 1.03))
    cov_ax.legend(handles=[Line2D([0], [0], color='.3', lw=1.8, label='All qualifying touch'),
                           Line2D([0], [0], color='.3', lw=1.8, ls='--', label='Frames with ≥2 fingertips')],
                  loc='lower right', frameon=False, fontsize=8)
    sustained_ax.set(title='Sustained contact', xticks=range(3), xticklabels=('≥2 overall', '≥2 while moving', '≥3 overall'),
                     ylim=(0, 1.06))
    sustained_ax.axhline(.5, color='.4', lw=1, ls=':')
    rhythm_ax.set(title='At least 3 fingertips · 5-second windows', xlabel='Simulated seconds', ylim=(0, 1.03))
    for ax in axes.ravel():
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=.22)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(root / 'contact-distribution.png', dpi=165)
    plt.close(fig)


def render_full(root):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from ..i04_geometric_traversal.fingertip_replay import show_split_tips
    from ..i04_geometric_traversal.pad_contact_visual import camera_from_direction, marker
    from .experiment import FingertipHand, rotation

    results = read_json(root / 'results.json')['runs']
    manifest = read_json(root / 'protocol.json')
    maximum_load = manifest['qualified_surface_region_load_n']
    if len(results) != 2:
        raise ValueError('The paired video requires both complete traces')
    if digest(root / 'model.mjb') != manifest['model_sha256']:
        raise ValueError('The replay model differs from the simulated model')
    source = (root / 'model.mjb').read_bytes()
    model = mujoco.MjModel.from_binary_path('pear.mjb', {'pear.mjb': source})
    states = []
    for r in results:
        case = root / r['arm']
        if digest(case / 'trace.npz') != r['trace_sha256']:
            raise ValueError(f'The recorded trajectory changed after scoring in {case}')
        with np.load(case / 'trace.npz', allow_pickle=False) as tape, np.load(case / 'coverage.npz', allow_pickle=False) as cov:
            states.append({k: tape[k] for k in ('qpos', 'points_m', 'pad_valid', 'mode', 'region_load_n')}
                          | {'coverage': cov['coverage_10_mm']})
    start = states[0]['qpos'][0]
    hand = FingertipHand(model, object_name='pear', grasp=(start[:3], rotation(start[3:7]), start[7:]),
                        min_head_fraction=None)
    data = [mujoco.MjData(model) for _ in states]
    show_split_tips(hand)
    tactile_gpis = manifest['profile'] == 'tactile-gpis-surface'
    width, height, bar, fps = 640, 400, 116 if tactile_gpis else 92, 24
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    model.vis.headlight.ambient[:] = .49
    model.vis.headlight.diffuse[:] = .85
    visual = mujoco.MjvOption()
    visual.geomgroup[3:] = 0
    camera = mujoco.MjvCamera()
    sx = 2 * width
    duration = max(r['seconds'] for r in results)
    speed = max(6, 2 * math.ceil(duration / 165.))
    opening_seconds = min(5., duration)
    opening_frames = int(round(opening_seconds * fps))
    frames = opening_frames + math.ceil((duration - opening_seconds) * fps / speed) + 1
    stem = f'{manifest["experiment"]}-full-pair'
    destination = root / f'{stem}.mp4'
    staged = root / f'{stem}.rendering.mp4'
    base_font = ImageFont.load_default(size=18)
    title_font = ImageFont.load_default(size=22)
    slides = [root / 'comparison.png']
    cooperative = manifest['profile'] in MULTIFINGER_PROFILES
    if cooperative:
        slides.insert(0, root / 'contact-distribution.png')
    if (root / 'signed-results.json').is_file():
        signed = read_json(root / 'signed-results.json')
        if signed['coverage_results_sha256'] != digest(root / 'results.json'):
            raise ValueError('The signed GPIS figure is from a different evaluated replay')
        slides.append(root / 'signed-comparison.png')
    if tactile_gpis and (root / 'signed-progression.png').is_file():
        slides.append(root / 'signed-progression.png')
    if tactile_gpis:
        from .active_planner import LONG_DISTAL

        distal_columns = hand.q[LONG_DISTAL]
    slide_frames = 6 * fps
    with mujoco.Renderer(model, height=height, width=width) as renderer, imageio.get_writer(
            staged, fps=fps, codec='libx264', quality=7, macro_block_size=2,
            ffmpeg_params=['-movflags', '+faststart']) as writer:
        for frame in range(frames):
            rate = 1 if frame < opening_frames else speed
            sim_t = frame / fps if rate == 1 else min(duration, opening_seconds + (frame - opening_frames) * rate / fps)
            canvas = Image.new('RGB', (sx, height + bar), '#12202a')
            azimuth = np.deg2rad(45 + 330 * frame / max(frames - 1, 1))
            look_from = np.array([np.cos(azimuth), np.sin(azimuth), .59])
            hud = []
            for panel, (d, trace, run) in enumerate(zip(data, states, results)):
                tick = min(int(round(sim_t / CONTROL_DT)), len(trace['qpos']) - 1)
                d.qpos[:] = trace['qpos'][tick]
                mujoco.mj_kinematics(model, d)
                mujoco.mj_camlight(model, d)
                distance = .77 + max(0., np.linalg.norm(d.qpos[:3]) - .20) * .64
                camera_from_direction(camera, look_from, d.qpos[:3] * .18, distance)
                renderer.update_scene(d, camera=camera, scene_option=visual)
                peak = float(trace['region_load_n'][tick].max())
                mark = (.10, 1., .37, 1.) if peak <= maximum_load else (1., .60, .10, 1.)
                for f in np.flatnonzero(trace['pad_valid'][tick]):
                    marker(renderer.scene, trace['points_m'][tick, f], .0036, mark)
                canvas.paste(Image.fromarray(renderer.render()), (panel * width, 0))
                phase = 'finished' if sim_t > run['seconds'] else trace['mode'][tick]
                touching = trace['pad_valid'][tick]
                distal = np.rad2deg(d.qpos[distal_columns]) if tactile_gpis else None
                hud.append((tick * CONTROL_DT, trace['coverage'][tick], phase, touching, peak, distal))
            draw = ImageDraw.Draw(canvas)
            if tactile_gpis:
                draw.text((15, 13), 'active-02-a  |  Online tactile GPIS', font=base_font, fill='#e9f1f3')
            for panel, (t, covered, phase, touching, peak, distal) in enumerate(hud):
                left = panel * width + 15
                draw.text((left, height + 9), DISPLAY_NAMES[panel], font=title_font, fill='white')
                draw.text((left, height + 39), f'{t:6.1f} s  |  {rate}x  |  {phase}  |  {touching.sum()}/4 finger contacts',
                          font=base_font, fill='#d3e9e4')
                draw.text((left, height + 63), f'Measured coverage 10 mm {covered:6.1%}   |   peak region {peak:5.1f} N',
                          font=base_font, fill='#b9ced5')
                if distal is not None:
                    angles = '  /  '.join(f'{angle:.0f}°' for angle in distal)
                    draw.text((left, height + 88), f'Long-finger distal flexion  {angles}',
                              font=base_font, fill='#b9ced5')
                if cooperative:
                    x = (panel + 1) * width - 143
                    for finger, (tag, active) in enumerate(zip(('I', 'M', 'R', 'T'), touching)):
                        left_box = x + 34 * finger
                        color = ('#38d18b' if peak <= maximum_load else '#e7a651') if active else '#394a58'
                        draw.rounded_rectangle((left_box, 14, left_box + 27, 40), radius=5, fill=color)
                        draw.text((left_box + 13, 27), tag, anchor='mm', font=base_font,
                                  fill='#18282b' if active else '#edf4f2')
            writer.append_data(np.asarray(canvas))
            if frame and frame % 250 == 0:
                print(f'Video {frame}/{frames} frames', flush=True)
        for slide in slides:
            with Image.open(slide) as comparison:
                panel = comparison.copy().convert('RGB')
            panel.thumbnail((sx, height + bar))
            tail = Image.new('RGB', (sx, height + bar), '#f4f7f5')
            tail.paste(panel, ((sx - panel.width) // 2, (height + bar - panel.height) // 2))
            still = np.asarray(tail)
            for _ in range(slide_frames):
                writer.append_data(still)
    staged.replace(destination)
    total_frames = frames + slide_frames * len(slides)
    write_json(root / 'video.json', dict(video=destination.name, sha256=digest(destination),
                                        frames=total_frames, fps=fps, playback_speed=speed,
                                        realtime_opening_s=opening_seconds,
                                        closing_figures=[slide.name for slide in slides],
                                        complete_time_span_sim_s=duration,
                                        trace_hashes={r['arm']: r['trace_sha256'] for r in results}))
    print(f'Video complete {destination.name}, {total_frames} frames at {fps} fps', flush=True)
