"""Score the measured contact area and R-GPIS meshes on an identical wrist tape."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import trimesh

from .contracts import ExperimentConfig
from .fingertip_hand import DT
from .fingertip_replay import chunks
from .fingertip_traversal import surface_graph, write_json
from .oracle import MeasuredCoverage


RADII_MM = (5, 10, 25)
ACCURACY_MM = (5, 10)
MESH_SAMPLES = 50000
METRIC_SEED = 408
REGION_LOAD_N = 8.


def read_json(case, file):
    return json.loads((case / file).read_bytes())


def head_witness(case):
    audit = read_json(case, 'native_audit.json')
    path = case / 'native_head_contacts.npz'
    if not audit['native_replay_matched'] or sha256(path.read_bytes()).hexdigest() != audit['head_contacts_sha256']:
        raise ValueError(f'Contact evidence does not match the native audit of {case}')
    with np.load(path, allow_pickle=False) as data:
        points, normals = data['point_m'], data['normal']
        step = data['step'].astype(np.int32)
        low_force = data['max_region_load_n'] < REGION_LOAD_N
        if len(step) != audit['head_observations'] or (step < 1).any() or (step > audit['states']).any():
            raise ValueError(f'Invalid contact frame indices in {path}')
    return points, normals, step, low_force, audit


def verified_wrist(case, dp_case, state_count):
    base, dp = (read_json(directory, 'manifest.json') for directory in (case, dp_case))
    for name in ('model_sha256', 'surface_sha256', 'dt_s', 'source_grasp', 'arm_joints'):
        if base[name] != dp[name]:
            raise ValueError(f'Fixed-wrist comparison setup differs: {name}')
    if not (case / 'initial.npz').read_bytes() == (dp_case / 'initial.npz').read_bytes():
        raise ValueError('The initial physical states differ')
    if base['controller']['parameters'] != dp['controller']['parameters']:
        raise ValueError('The two cases must share MCC parameters')
    wrist = sha256()
    states = 0
    phases = Counter()
    for left, right in zip(chunks(case), chunks(dp_case), strict=True):
        np.testing.assert_array_equal(left['palm_command'], right['palm_command'])
        np.testing.assert_array_equal(left['phase'], right['phase'])
        wrist.update(np.ascontiguousarray(left['palm_command'], dtype='<f8').tobytes())
        names, counts = np.unique(left['phase'], return_counts=True)
        phases.update(dict(zip(names, counts)))
        states += len(left['palm_command'])
    if states != state_count or states != dp['wrist_steps'] or wrist.hexdigest() != dp['wrist_sha256']:
        raise ValueError('The compared wrist tape differs from its manifest')
    return wrist.hexdigest(), {name: int(count) * DT for name, count in sorted(phases.items())}


def planning_profile(case, profile):
    for file in ('initial.npz', 'trace_chunks.json', 'planned_patches.json'):
        if (case / file).read_bytes() != (profile / file).read_bytes():
            raise ValueError(f'Planning profile failed the original trajectory comparison: {file}')
    if read_json(case, 'manifest.json')['model_sha256'] != read_json(profile, 'manifest.json')['model_sha256']:
        raise ValueError('Planning profile uses a different physics model')
    return read_json(profile, 'summary.json')['timing']


def coverage_score(ledger, points, normals, step, low_force, states, sample_steps):
    graph, config, vertex_area = ledger.graph, ledger.config, ledger.vertex_area
    first = np.full((2, len(vertex_area)), states + 1, dtype=np.int32)
    registered = 0
    for start in range(0, len(points), 12000):
        batch = slice(start, start + 12000)
        xyz, normal = points[batch], normals[batch]
        vertex, error = graph.oriented_nearest_vertices(xyz, normal)
        cosine = np.einsum('ij,ij->i', graph.normals[vertex], normal)
        valid = (error <= config.mesh_residual_limit_m) & (cosine >= config.normal_alignment_cos)
        registered += int(valid.sum())
        np.minimum.at(first[0], vertex[valid], step[batch][valid])
        valid = valid & low_force[batch]
        np.minimum.at(first[1], vertex[valid], step[batch][valid])

    metrics, masks = {}, {}
    for name, directly_observed in zip(('all_heads', 'region_below_8n'), first):
        sources = np.flatnonzero(directly_observed <= states)
        masks[f'{name}_registered'] = directly_observed <= states
        distances = np.full(len(vertex_area), np.inf)
        if len(sources):
            distances = dijkstra(graph.adjacency, directed=False, indices=sources, min_only=True,
                                 limit=config.coverage_radius_m)
        masks[name] = distances <= config.coverage_radius_m
        earliest = np.full_like(directly_observed, states + 1)
        for vertex in sources:
            neighbors = ledger.neighborhood(int(vertex))
            earliest[neighbors] = np.minimum(earliest[neighbors], directly_observed[vertex])
        np.testing.assert_array_equal(earliest <= states, masks[name])
        masks[f'{name}_first_step'] = earliest
        metrics[name] = dict(area_fraction_by_radius_mm={str(r): min(1.,
            float(vertex_area[distances <= r * .001].sum() / vertex_area.sum())) for r in RADII_MM},
            area_fraction_over_time_25mm=np.clip((earliest <= sample_steps[:, None]) @ vertex_area /
                                                vertex_area.sum(), 0., 1.).tolist(),
            covered_vertices_25mm=int(masks[name].sum()),
            directly_observed_mesh_vertices=len(sources))
    quorum_by_step = np.bincount(step, minlength=states + 1)[1:]
    return dict(registered_contact_rows=registered, measured_contact_rows=len(points),
                at_least_one_head_fraction=float(np.mean(quorum_by_step >= 1)),
                four_heads_fraction=float(np.mean(quorum_by_step == 4)), **metrics), masks


def mesh_accuracy(mesh, gt_cloud):
    samples, _ = trimesh.sample.sample_surface(mesh, MESH_SAMPLES, seed=METRIC_SEED)
    recon_to_gt = cKDTree(gt_cloud).query(samples)[0] * 1000.
    gt_to_recon = cKDTree(samples).query(gt_cloud)[0] * 1000.
    result = dict(chamfer_mm2=float(np.mean(recon_to_gt**2) + np.mean(gt_to_recon**2)),
                  symmetric_mean_mm=float(.5 * (recon_to_gt.mean() + gt_to_recon.mean())),
                  recon_to_truth_mean_mm=float(recon_to_gt.mean()),
                  recon_to_truth_p95_mm=float(np.percentile(recon_to_gt, 95)),
                  truth_to_recon_mean_mm=float(gt_to_recon.mean()),
                  truth_to_recon_p95_mm=float(np.percentile(gt_to_recon, 95)),
                  at_mm={})
    for threshold in ACCURACY_MM:
        precision = float(np.mean(recon_to_gt <= threshold))
        recall = float(np.mean(gt_to_recon <= threshold))
        fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        result['at_mm'][str(threshold)] = dict(precision=precision, recall=recall, fscore=fscore)
    return result


def render_summary(directory, seconds, branches, graph, meshes, gt_cloud, dp_label='Finger DP + MCC',
                   mcc_label='Joint planning + MCC', *, coverage_limit=105, legend_location='lower right',
                   case_labels=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, to_rgb
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    colors = ('#087f74', '#ad591c', '#466193')
    labels = (mcc_label, dp_label) if case_labels is None else case_labels
    if not len(labels) == len(branches) == len(meshes) <= len(colors):
        raise ValueError('Reconstruction cases, meshes and labels must match')
    columns = len(labels) + 1
    fig = plt.figure(figsize=(4.33 * columns, 9.2), facecolor='white', layout='constrained')
    grid = fig.add_gridspec(2, columns, height_ratios=[1., 1.1])
    curves = fig.add_subplot(grid[0, 0])
    for label, color, data in zip(labels, colors, branches):
        full = np.asarray(data['coverage']['all_heads']['area_fraction_over_time_25mm']) * 100.
        low = np.asarray(data['coverage']['region_below_8n']['area_fraction_over_time_25mm']) * 100.
        curves.plot(seconds, full, color=color, linewidth=2.3, label=label)
        curves.plot(seconds, low, '--', color=color, linewidth=1.25)
    curves.set(xlabel='Simulated time (s)', ylabel='Area covered (%)', ylim=(0, coverage_limit),
               title='Measured surface coverage · 25 mm')
    curves.grid(alpha=.19)
    handles, _ = curves.get_legend_handles_labels()
    handles.append(Line2D([], [], color='#667077', linestyle='--', label='All regions below 8 N'))
    curves.legend(handles=handles, loc=legend_location, frameon=False, fontsize=9)
    figures = [mesh for mesh in (graph.mesh, *meshes) if mesh is not None]

    def show_xyz(points):
        return points[..., [0, 2, 1]] * [1000., 1000., -1000.]

    vertices = [show_xyz(mesh.vertices) for mesh in figures]
    all_vertices = np.vstack(vertices)
    low, high = all_vertices.min(axis=0) - 8, all_vertices.max(axis=0) + 8

    def view(cell, title):
        ax = fig.add_subplot(cell, projection='3d')
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
        ax.set_box_aspect(high - low, zoom=1.18)
        ax.view_init(elev=16, azim=-60)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, fontweight='medium')
        return ax

    for column, (data, label, color) in enumerate(zip(branches, labels, colors), 1):
        ax = view(grid[0, column], label + '\nMeasured head contacts')
        ax.add_collection3d(Poly3DCollection(vertices[0][graph.mesh.faces], edgecolor='none',
                                            facecolor='#8091a1', alpha=.11))
        points = show_xyz(data['voxels'])
        ax.scatter(*points.T, s=2.7, color=color, alpha=.78, depthshade=False)

    distance_map = ScalarMappable(norm=Normalize(0, 20), cmap='viridis')
    gt_tree = cKDTree(gt_cloud)
    axes = []
    for column, mesh in enumerate((graph.mesh, *meshes)):
        if column == 0:
            title = 'Reference surface'
            display_normals = mesh.face_normals[:, [0, 2, 1]] * [1., 1., -1.]
            light = .50 + .50 * np.maximum(0., display_normals @ [.42, -.48, .76])
            color = light[:, None] * to_rgb('#b7c9d2')
        else:
            if mesh is None:
                view(grid[1, column], f'{labels[column - 1]}\nNo reconstructed surface')
                continue
            score = branches[column - 1]['reconstruction']['accuracy']['at_mm']['5']['fscore']
            title = f'{labels[column - 1]}\nR-GPIS · F-score@5 mm {score:.1%}'
            distance = gt_tree.query(mesh.triangles_center)[0] * 1000.
            color = distance_map.to_rgba(distance)
        ax = view(grid[1, column], title)
        xyz = show_xyz(mesh.vertices)
        ax.add_collection3d(Poly3DCollection(xyz[mesh.faces], facecolors=color, edgecolors='none'))
        axes.append(ax)
    colorbar = fig.colorbar(distance_map, ax=axes, location='bottom', shrink=.58, pad=.035, aspect=35,
                           extend='max')
    colorbar.set_label('Reconstructed surface to reference distance (mm)')
    fig.savefig(directory / 'surface_results.png', dpi=165)
    plt.close(fig)


def compare(case, dp_case, profile_case, output):
    import torch
    from . import fingertip_r_gpis as rgpis

    torch.set_num_threads(1)
    base, dp = head_witness(case), head_witness(dp_case)
    states = base[4]['states']
    if states != dp[4]['states']:
        raise ValueError('Report requires two complete, equal-length wrist recordings')
    wrist_hash, phase_time = verified_wrist(case, dp_case, states)
    planning = planning_profile(case, profile_case)
    graph = surface_graph()
    config = ExperimentConfig()
    ledger = MeasuredCoverage(graph, config)
    gt_cloud, _ = trimesh.sample.sample_surface(graph.mesh, MESH_SAMPLES, seed=METRIC_SEED + 1)
    dp_summary = read_json(dp_case, 'summary.json')
    dp_trip = dp_summary['first_sustained_overload_stop']
    sample_steps = np.r_[np.arange(0, states, int(10 / DT)), states]
    if dp_trip:
        sample_steps = np.union1d(sample_steps, [dp_trip['step']])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(profile_case / 'planning_timing.npz', output / 'planning_timing.npz')
    eval_bounds = rgpis.bounds_for_contacts(np.vstack((base[0], dp[0])))
    branches, meshes = [], []
    for name, (points, normals, step, low_force, audit) in zip(('joint_mcc', 'finger_dp_mcc'), (base, dp)):
        coverage, masks = coverage_score(ledger, points, normals, step, low_force, states, sample_steps)
        print(f'{name}: measured 25 mm coverage {coverage["all_heads"]["area_fraction_by_radius_mm"]["25"]:.2%}',
              flush=True)
        voxel_indices = rgpis.voxel_representatives(points)
        selected = voxel_indices[rgpis.farthest_indices(points[voxel_indices], rgpis.CONTACT_BUDGET)]
        training, training_normals = points[selected].astype(np.float64), normals[selected].astype(np.float64)
        voxels = points[voxel_indices]
        mesh, reconstruction = rgpis.reconstruct(training, training_normals,
                                                   rgpis.bounds_for_contacts(points), eval_bounds)
        reconstruction['accuracy'] = mesh_accuracy(mesh, gt_cloud)
        reconstruction['occupied_input_voxels'] = len(voxels)
        mesh.export(output / f'{name}_r_gpis.ply')
        np.savez_compressed(output / f'{name}_surface_evidence.npz', training_contacts_m=training,
                            training_normals=training_normals, training_steps=step[selected],
                            vertex_area_m2=ledger.vertex_area,
                            head_contacts_sha256=audit['head_contacts_sha256'], **masks)
        diagnostics = dict(outside_grasp_fraction=audit['illegal_grasp_states'] / states,
            incidental_fraction=audit['incidental_contact_steps'] / states,
            load_excluded_fraction=audit['force_excluded_states'] / states,
            peak_region_load_n=max(audit['peak_load_n_fingers_then_palm']),
            longest_over_40n_s=audit['longest_over_40n_s'],
            native_audit_max_qpos_error=audit['max_qpos_error'],
            head_contacts_sha256=audit['head_contacts_sha256'])
        branches.append(dict(name=name, coverage=coverage, reconstruction=reconstruction, diagnostics=diagnostics,
                             voxels=voxels))
        meshes.append(mesh)
    summary = read_json(case, 'summary.json')
    metrics_only = [{key: value for key, value in branch.items() if key != 'voxels'}
                    for branch in branches]
    result = dict(experiment='I04-c continuous surface and fixed-wrist Finger DP', wrist_sha256=wrist_hash,
        states=states, executed_s=states * DT, base=case.name, dp=dp_case.name,
        sample_times_s=(sample_steps * DT).tolist(),
        protocol=dict(contact_definition='measured distal head >= 0.2 N; no target or pose gate',
            pose_gate=False, waypoint_gate=False, mesh_residual_limit_m=config.mesh_residual_limit_m,
            cosine_min=config.normal_alignment_cos, radii_mm=RADII_MM, secondary_region_load_cutoff_n=REGION_LOAD_N,
            mesh_samples_per_surface=MESH_SAMPLES, mesh_sampling_seeds=[METRIC_SEED + 1, METRIC_SEED],
            r_gpis=dict(reference=rgpis.REFERENCE, voxel_m=rgpis.VOXEL_M, contact_budget=rgpis.CONTACT_BUDGET,
                coordinate_unit_m=rgpis.COORDINATE_UNIT_M, dummy_grid=rgpis.DUMMY_GRID,
                nearest_grid_points_removed=rgpis.REMOVE_NEIGHBORS, bounds_padding_fraction=rgpis.BOX_MARGIN_FRACTION,
                optimizer='Adam', iterations=rgpis.TRAINING_ITERATIONS, lr=rgpis.LEARNING_RATE,
                grid_step_limit_m=rgpis.GRID_STEP_M, extraction='posterior maxima along observed head normals',
                minimum_ridge_response=rgpis.MIN_RESPONSE, maximum_contact_support_m=rgpis.CONTACT_SUPPORT_M,
                normal_neighbors=rgpis.NORMAL_NEIGHBORS,
                evaluation_bounds_m=eval_bounds.tolist(), torch=torch.__version__, device='CPU, one thread')),
        continuous=dict(routes=summary['executed_routes_by_mode'], phases_s=phase_time, lifts=summary['lifts'],
                        waypoints=summary['surface_waypoints']),
        timings=dict(joint_planning=planning, finger_dp=dp_summary['timing'],
                     dp_native_wall_s=dp_summary['native_wall_s']),
        dp_scheduling=dp_summary.get('scheduling'),
        dp_first_sustained_high_force=dp_trip,
        branches=metrics_only)
    write_json(output / 'surface_comparison.json', result)
    mode = dp_summary.get('scheduling', {}).get('mode')
    label_suffix = {'rolling-16-8': ' · 16/8', 'async-100ms': ' · async 100 ms'}.get(mode, '')
    dp_label = 'Finger DP + MCC' + label_suffix
    render_summary(output, sample_steps * DT, branches, graph, meshes, gt_cloud, dp_label)
    return result


def add_arguments(parser):
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--dp-case', type=Path, required=True)
    parser.add_argument('--planning-case', type=Path)
    parser.add_argument('--output', type=Path, required=True)
