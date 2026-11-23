"""Score acitive-02 contacts and common R-GPIS reconstructions against the pear."""

from concurrent.futures import ProcessPoolExecutor
from functools import cache
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import trimesh

from .experiment import ARMS, CONTROL_DT, EXPERIMENT, PEAR, QUALITY, digest, read_json, write_json
from ..i04_geometric_traversal import fingertip_r_gpis as rgpis
from ..i04_geometric_traversal.contracts import ExperimentConfig
from ..i04_geometric_traversal.fingertip_surface_report import MESH_SAMPLES, METRIC_SEED, mesh_accuracy
from ..i04_geometric_traversal.fixtures import SurfaceGraph
from ..i04_geometric_traversal.oracle import MeasuredCoverage

AREA_RADIUS_M = .010
DISPLAY_NAMES = ('Aude + MCC', 'Aude + single_all + MCC')
COLORS = ('#176e77', '#ba552a')


def registered_vertices(ledger, points, normals):
    graph = ledger.graph
    vertices, error = graph.oriented_nearest_vertices(points, normals)
    normal_cos = np.einsum('ij,ij->i', graph.normals[vertices], normals)
    registered = (error <= ledger.config.mesh_residual_limit_m) & (normal_cos >= ledger.config.normal_alignment_cos)
    return vertices, registered


def vertex_coverage(ledger, vertices, ticks, total_ticks, *, return_witness=False):
    first_witness = np.full(len(ledger.graph.vertices_m), total_ticks + 1, dtype=int)
    np.minimum.at(first_witness, vertices, ticks)
    first_covered = first_witness.copy()
    for vertex in np.flatnonzero(first_witness <= total_ticks):
        neighbors = ledger.neighborhood(int(vertex))
        np.minimum.at(first_covered, neighbors, first_witness[vertex])
    weights = ledger.vertex_area / ledger.vertex_area.sum()
    by_tick = np.bincount(first_covered, weights=weights, minlength=total_ticks + 2)
    result = (np.cumsum(by_tick[:total_ticks + 1]).clip(0., 1.), len(vertices))
    return (*result, first_covered) if return_witness else result


def coverage(ledger, points, normals, ticks, total_ticks, *, return_witness=False):
    vertices, keep = registered_vertices(ledger, points, normals)
    return vertex_coverage(ledger, vertices[keep], ticks[keep], total_ticks, return_witness=return_witness)


def reconstruct(case, tick, points, normals, truth_cloud):
    started = perf_counter()
    reps = rgpis.voxel_representatives(points) if len(points) else np.empty(0, dtype=int)
    reps = reps[rgpis.farthest_indices(points[reps], rgpis.CONTACT_BUDGET)]
    path = case / f'r-gpis-{tick:05d}.ply'
    result = dict(fscore_5_mm=0., precision_5_mm=0., recall_5_mm=0., symmetric_mean_mm=None,
                  reconstruction_voxels=len(reps), mesh=None, mesh_written_wall_s=None)
    if len(reps) < 4:
        status = 'insufficient_voxels'
    else:
        support = points[reps]
        # Query the contact support plus the existing ridge cutoff. No truth
        # dimensions or normals are given to the reconstruction.
        extent = rgpis.CONTACT_SUPPORT_M + 2 * rgpis.GRID_STEP_M
        box = np.vstack((support.min(axis=0) - extent, support.max(axis=0) + extent))
        try:
            mesh, _ = rgpis.reconstruct(support, normals[reps], rgpis.bounds_for_contacts(support), box)
        except ValueError as exc:
            status = str(exc)
        except RuntimeError as exc:
            if str(exc) != 'No surface found at the given iso value.':
                raise
            status = 'no_supported_surface'
        else:
            mesh.export(path)
            mesh_wall_s = perf_counter() - started
            quality = mesh_accuracy(mesh, truth_cloud)
            scores = quality['at_mm']['5']
            result.update(fscore_5_mm=scores['fscore'], precision_5_mm=scores['precision'],
                          recall_5_mm=scores['recall'], symmetric_mean_mm=quality['symmetric_mean_mm'],
                          mesh=path.name, mesh_written_wall_s=mesh_wall_s,
                          mesh_vertices=len(mesh.vertices), mesh_triangles=len(mesh.faces),
                          mesh_watertight=bool(mesh.is_watertight))
            status = 'mesh'
    return dict(**result, reconstruction=status, reconstruction_wall_s=perf_counter() - started)


def score_arm(case, run, ledger, truth_cloud, readouts):
    score_path = case / 'scores.json'
    trace_path = case / 'trace.npz'
    trace_hash = digest(trace_path)
    scorer_hash = digest(Path(__file__))
    if score_path.is_file():
        prior = read_json(score_path)
        if (prior.get('trace_sha256') == trace_hash and prior.get('scoring_source_sha256') == scorer_hash
                and [row['comparison_time_s'] for row in prior['curve']] == readouts
                and all(row['mesh'] is None or (case / row['mesh']).is_file() for row in prior['curve'])):
            return prior
    with np.load(trace_path, allow_pickle=False) as trace:
        ticks, fingers = np.nonzero(trace['pad_valid'])
        points = trace['points_m'][ticks, fingers]
        normals = trace['normals_outward'][ticks, fingers]
        total_ticks = len(trace['qpos']) - 1
        wall_s = trace['wall_s']
        error = np.max(abs(trace['finger_nominal'] - trace['aude_joint_nominal']))
        if run['arm'] == ARMS[0] and error != 0:
            raise ValueError(f'Aude joint references were not dispatched in {case}')
        if run['arm'] == ARMS[1] and error <= 1e-7:
            raise ValueError(f'Finger Policy did not take command authority in {case}')
        if 'wrist_offset' in trace:
            np.testing.assert_allclose(trace['wrist_command'][1:, :3] - trace['wrist_nominal'][1:, :3],
                                       trace['wrist_offset'][1:, :3], rtol=0., atol=1e-9)
    area_curve, registered = coverage(ledger, points, normals, ticks, total_ticks)
    hit = np.flatnonzero(area_curve >= QUALITY['coverage_10_mm'])
    curves, meshes = [], {}
    reconstruction_s = 0.
    for time_s in readouts:
        at = min(int(round(time_s / CONTROL_DT)), total_ticks)
        if at not in meshes:
            keep = ticks <= at
            meshes[at] = reconstruct(case, at, points[keep], normals[keep], truth_cloud)
            reconstruction_s += meshes[at]['reconstruction_wall_s']
        curves.append(dict(comparison_time_s=time_s, observed_time_s=at * CONTROL_DT,
                           coverage_10_mm=float(area_curve[at]), **meshes[at]))
    final = curves[-1]
    result = dict(**run, case=case.name, registered_contacts=registered, total_contacts=len(points),
                  trace_sha256=trace_hash, scoring_source_sha256=scorer_hash,
                  coverage_10_mm=final['coverage_10_mm'],
                  new_coverage_10_mm=final['coverage_10_mm'] - curves[0]['coverage_10_mm'],
                  fscore_5_mm=final['fscore_5_mm'], precision_5_mm=final['precision_5_mm'],
                  recall_5_mm=final['recall_5_mm'], symmetric_mean_mm=final['symmetric_mean_mm'],
                  completed=run['stop'] == 'budget_complete',
                  quality_met=(run['stop'] == 'budget_complete' and
                               final['fscore_5_mm'] >= QUALITY['fscore_5_mm'] and len(hit) > 0),
                  time_to_70_coverage_sim_s=float(hit[0] * CONTROL_DT) if len(hit) else None,
                  time_to_70_coverage_wall_s=float(wall_s[hit[0]]) if len(hit) else None,
                  final_mesh=final['mesh'], reconstruction_wall_s=reconstruction_s, curve=curves)
    write_json(score_path, result)
    return result


def verify_pair(pair, results):
    if results[0]['initial_state_sha256'] != results[1]['initial_state_sha256']:
        raise ValueError(f'The arms started from different physical states in {pair}')
    plans = []
    for arm in ARMS:
        with (pair / arm / 'plans.jsonl').open(encoding='utf-8') as stream:
            plans.append(json.loads(stream.readline()))
    for key in ('wrist_target', 'aude_joint_target', 'virtual_contacts_m'):
        np.testing.assert_allclose(plans[0][key], plans[1][key], rtol=0., atol=1e-7, err_msg=str(pair))
    return True


def summarize(results, arm):
    runs = [x for x in results if x['arm'] == arm]
    keys = ('coverage_10_mm', 'new_coverage_10_mm', 'fscore_5_mm', 'precision_5_mm', 'recall_5_mm',
            'symmetric_mean_mm', 'at_least_one_contact', 'at_least_three_contacts', 'four_contacts',
            'longest_full_loss_s', 'wrist_translation_m', 'wrist_rotation_deg', 'max_region_load_n',
            'seconds', 'wall_execution_s', 'time_to_70_coverage_sim_s', 'time_to_70_coverage_wall_s',
            'max_palm_load_n', 'p95_palm_contact_load_n', 'wrist_mcc_translation_max_m',
            'wrist_mcc_yield_time_s', 'palm_proximity_yield_time_s', 'longest_above_40n_s')
    metrics = {}
    for key in keys:
        values = [x[key] for x in runs if x.get(key) is not None]
        if values:
            median, low, high = np.percentile(values, [50, 0, 100])
            metrics[key] = dict(median=float(median), min=float(low), max=float(high), n=len(values))
        else:
            metrics[key] = None
    return dict(n=len(runs), completed=sum(x['completed'] for x in runs),
                quality_met=sum(x['quality_met'] for x in runs),
                reached_70_coverage=sum(x['time_to_70_coverage_sim_s'] is not None for x in runs), **metrics)


def chart(output, truth_cloud, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig = plt.figure(figsize=(12, 8.4), layout='constrained', facecolor='#f3f6f5')
    layout = fig.add_gridspec(2, 2, height_ratios=[.86, 1.14])
    fig.suptitle('acitive-02  /  YCB 016 Pear  /  Shared wrist MCC', x=.015, ha='left',
                 fontsize=17, fontweight='bold')
    for i, (key, title) in enumerate((('coverage_10_mm', 'Measured geodesic coverage at 10 mm'),
                                      ('fscore_5_mm', 'Reconstruction F-score at 5 mm'))):
        ax = fig.add_subplot(layout[0, i])
        high_water = QUALITY[key]
        for arm, label, color in zip(ARMS, DISPLAY_NAMES, COLORS):
            trials = [r['curve'] for r in results if r['arm'] == arm]
            x = [c['comparison_time_s'] for c in trials[0]]
            values = np.array([[c[key] for c in curve] for curve in trials])
            low, median, high = np.percentile(values, [25, 50, 75], axis=0)
            high_water = max(high_water, float(high.max()))
            ax.fill_between(x, low, high, color=color, alpha=.16)
            ax.plot(x, median, 'o-', label=label, color=color, linewidth=2.2, markersize=4)
        ax.axhline(QUALITY[key], color='#7e8c89', linewidth=.9, linestyle='--')
        ax.set(title=title, xlabel='Simulated exploration time (s)', ylim=(0., min(1., high_water + .08)))
        ax.yaxis.set_major_formatter(PercentFormatter(1.))
        ax.grid(axis='y', alpha=.23)
        ax.spines[['right', 'top']].set_visible(False)
        if i == 0:
            ax.legend(frameon=False, loc='upper left', fontsize=9)
    frame = results[0]['start_frame']
    halfspan = np.ptp(truth_cloud, axis=0) / 2. + .025
    center = (truth_cloud.max(axis=0) + truth_cloud.min(axis=0)) / 2.
    for i, (arm, label, color) in enumerate(zip(ARMS, DISPLAY_NAMES, COLORS)):
        result = next(r for r in results if r['start_frame'] == frame and r['arm'] == arm)
        ax = fig.add_subplot(layout[1, i], projection='3d', facecolor='#f3f6f5')
        ax.scatter(*truth_cloud[::13].T, s=1.1, c='#8a9a97', alpha=.13, depthshade=False)
        if name := result['final_mesh']:
            mesh = trimesh.load(output / f'frame-{frame:05d}' / arm / name, force='mesh', process=False)
            surface, _ = trimesh.sample.sample_surface(mesh, 3600, seed=METRIC_SEED)
            ax.scatter(*surface.T, s=2.5, c=color, alpha=.76, depthshade=False)
        ax.set_xlim(*(center[0] + np.array([-1, 1]) * halfspan[0]))
        ax.set_ylim(*(center[1] + np.array([-1, 1]) * halfspan[1]))
        ax.set_zlim(*(center[2] + np.array([-1, 1]) * halfspan[2]))
        ax.set_box_aspect(halfspan.copy())
        ax.view_init(elev=24, azim=45)
        ax.set_axis_off()
        ax.set_title(f'{label}\nFirst pair, frame {frame}   |   F@5 mm {result["fscore_5_mm"]:.1%}', fontsize=10)
    fig.savefig(output / 'comparison.png', dpi=180)
    plt.close(fig)


@cache
def evaluation_reference():
    torch.set_num_threads(1)
    truth = trimesh.load(PEAR, force='mesh', process=False)
    cloud, _ = trimesh.sample.sample_surface(truth, MESH_SAMPLES, seed=METRIC_SEED + 1)
    graph = SurfaceGraph(truth, coverage_radius_m=.025)
    return cloud, MeasuredCoverage(graph, ExperimentConfig(coverage_radius_m=AREA_RADIUS_M))


def score_pair(job):
    output, frame, current, readouts = job
    truth_cloud, ledger = evaluation_reference()
    pair = output / f'frame-{frame:05d}'
    verify_pair(pair, current)
    scored = [score_arm(pair / arm, run, ledger, truth_cloud, readouts) for arm, run in zip(ARMS, current)]
    baseline, dp = scored
    paired = dict(start_frame=frame, identical_first_aude_plan=True,
                  **{f'delta_{key}': dp[key] - baseline[key] for key in
                     ('coverage_10_mm', 'fscore_5_mm', 'at_least_three_contacts', 'longest_full_loss_s')})
    print(f'Scored {frame}: F@5 mm {baseline["fscore_5_mm"]:.3f} / {dp["fscore_5_mm"]:.3f}, '
          f'coverage@10 mm {baseline["coverage_10_mm"]:.3f} / {dp["coverage_10_mm"]:.3f}', flush=True)
    return scored, paired


def score_suite(output: Path):
    protocol = read_json(output / 'protocol.json')
    runs = read_json(output / 'runs.json')
    got = {(r['start_frame'], r['arm']) for r in runs}
    frames = [f for f in protocol['tried_start_frames'] if all((f, arm) in got for arm in ARMS)]
    if not frames:
        raise ValueError('Scoring requires at least one complete pair of execution records')
    budget = protocol['seconds']
    readouts = [0., *[t for t in (30., 60., 120.) if t < budget], budget]
    all_runs = {(r['start_frame'], r['arm']): r for r in runs}
    jobs = [(output, frame, [all_runs[frame, arm] for arm in ARMS], readouts) for frame in frames]
    workers = min(4, len(frames))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pairs = list(pool.map(score_pair, jobs))
    else:
        pairs = [score_pair(jobs[0])]
    results = [result for scored, _ in pairs for result in scored]
    paired = [comparison for _, comparison in pairs]
    summary = dict(experiment=EXPERIMENT, paired_start_frames=frames,
                   planned_pairs=len(protocol['tried_start_frames']), scored_pairs=len(frames),
                   budget_s=budget, scoring_workers=workers,
                   quality_thresholds=QUALITY, outcome_method='same offline R-GPIS using actual pad taps',
                   scoring_source_sha256=digest(Path(__file__)),
                   methods={arm: summarize(results, arm) for arm in ARMS}, trials=results, differences=paired)
    write_json(output / 'results.json', summary)
    truth_cloud, _ = evaluation_reference()
    chart(output, truth_cloud, results)
