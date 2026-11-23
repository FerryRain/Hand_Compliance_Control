"""Reconstruct the paired tactile traces with an oriented signed GPIS."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import gpytorch
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from skimage.measure import marching_cubes
import torch
import trimesh

from .experiment import ARMS, CONTROL_DT, digest, read_json, write_json
from .report import COLORS, DISPLAY_NAMES, evaluation_reference
from ..i04_geometric_traversal import fingertip_r_gpis as rgpis
from ..i04_geometric_traversal.fingertip_surface_report import mesh_accuracy


NORMAL_OFFSET_M = .006
LENGTHSCALE_SPACING_MULTIPLE = 2.6
LENGTHSCALE_BOUNDS_M = (.010, .065)
NOISE_VARIANCE = .003
MIN_FACE_NORMAL_COSINE = .42


def cpu_posterior(train, labels, length):
    kernel = gpytorch.kernels.MaternKernel(nu=1.5).double()
    with torch.inference_mode():
        kernel.lengthscale = length
        t = torch.from_numpy(train)
        cov = kernel(t, t).to_dense().numpy().copy()
    cov.flat[::len(cov) + 1] += NOISE_VARIANCE
    weights = cho_solve(cho_factor(cov, overwrite_a=True), labels)
    gain = np.sqrt(3.) / length

    def predict(points):
        scaled = cdist(points, train) * gain
        return ((1. + scaled) * np.exp(-scaled)) @ weights

    return predict


def signed_gpis(contacts, normals, *, device="cpu"):
    support = cKDTree(contacts)
    distance, _ = support.query(contacts, k=2)
    length = float(np.clip(LENGTHSCALE_SPACING_MULTIPLE * np.median(distance[:, 1]), *LENGTHSCALE_BOUNDS_M))
    train = np.vstack((contacts, contacts - NORMAL_OFFSET_M * normals, contacts + NORMAL_OFFSET_M * normals))
    labels = np.repeat((0., -1., 1.), len(contacts))
    target = torch.device(device)
    if target.type == "cuda":
        from .signed_cuda import cuda_posterior

        predict = cuda_posterior(train, labels, length, NOISE_VARIANCE, target)
    elif target.type == "cpu":
        predict = cpu_posterior(train, labels, length)
    else:
        raise ValueError(f"Unsupported signed GPIS device: {target}")

    limit = rgpis.CONTACT_SUPPORT_M
    grid = rgpis.GRID_STEP_M
    query_margin = limit + 2 * grid
    origin = contacts.min(axis=0) - query_margin
    extent = contacts.max(axis=0) + query_margin - origin
    shape = np.ceil(extent / grid).astype(int) + 1
    spacing = extent / (shape - 1)
    field = np.zeros(np.prod(shape), dtype=np.float32)
    mask = np.zeros(len(field), dtype=bool)
    for start in range(0, len(field), rgpis.QUERY_BATCH):
        end = min(start + rgpis.QUERY_BATCH, len(field))
        at = np.stack(np.unravel_index(np.arange(start, end), shape), axis=-1) * spacing + origin
        pad_distance, _ = support.query(at)
        nearby = pad_distance <= query_margin
        if nearby.any():
            local = field[start:end]
            local[nearby] = predict(at[nearby])
        mask[start:end] = pad_distance <= limit
    vertices, faces, _, _ = marching_cubes(field.reshape(shape), level=0., spacing=spacing,
                                           mask=mask.reshape(shape), allow_degenerate=False)
    mesh = trimesh.Trimesh(vertices + origin, faces, process=True)
    distance, nearest = support.query(mesh.triangles_center)
    agreement = np.einsum('ij,ij->i', mesh.face_normals, normals[nearest])
    keep = (distance <= limit) & (abs(agreement) > MIN_FACE_NORMAL_COSINE)
    if not keep.any():
        raise ValueError('No supported signed zero level')
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    if np.median(agreement[keep]) < 0.:
        mesh.invert()
    return mesh, length


def signed_output_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def signed_surface_for_output(contacts, normals, *, device=None):
    target = signed_output_device() if device is None else str(torch.device(device))
    try:
        mesh, length = signed_gpis(contacts, normals, device=target)
    except torch.cuda.OutOfMemoryError:
        if device is not None or target != 'cuda':
            raise
        target = 'cpu'
        mesh, length = signed_gpis(contacts, normals, device=target)
    return mesh, length, target


def evaluate(job):
    root, run, maximum_load = job
    arm = run['arm']
    case = root / arm
    truth, _ = evaluation_reference()
    trace = case / 'trace.npz'
    if digest(trace) != run['trace_sha256']:
        raise ValueError(f'The signed GPIS input changed since the shared baseline scoring in {case}')
    with np.load(trace, allow_pickle=False) as data:
        qualified = data['pad_valid'] & (data['region_load_n'].max(axis=1) <= maximum_load)[:, None]
        tick, finger = np.nonzero(qualified)
        points, normals = data['points_m'][tick, finger], data['normals_outward'][tick, finger]
    rows = run['snapshots']
    scores = []
    final_path = case / 'gpis-signed-final.ply'
    for i in (len(rows) - 1, *range(len(rows) - 1)):
        source = rows[i]
        started = perf_counter()
        visible = tick <= round(source['time_s'] / CONTROL_DT)
        observed, orientation = points[visible], normals[visible]
        representatives = rgpis.voxel_representatives(observed) if len(observed) else np.empty(0, dtype=int)
        reps = representatives[rgpis.farthest_indices(observed[representatives], rgpis.CONTACT_BUDGET)]
        metrics = dict(time_s=source['time_s'], coverage_10_mm=source['coverage_10_mm'],
                       reconstruction_voxels=len(reps), fscore_5_mm=0., symmetric_mean_mm=None, mesh=None)
        if len(reps) >= 4:
            try:
                mesh, length, backend = signed_surface_for_output(observed[reps], orientation[reps])
                metrics['gpis_device'] = backend
            except (ValueError, RuntimeError) as exc:
                if str(exc) not in ('No surface found at the given iso value.', 'No supported signed zero level',
                                    'Surface level must be within volume data range.'):
                    raise
                status = 'no_supported_zero_level'
            else:
                if i == len(rows) - 1 or run['experiment'] == 'active-02-a':
                    path = final_path if i == len(rows) - 1 else case / f'gpis-signed-{round(source["time_s"] / CONTROL_DT):05d}.ply'
                    mesh.export(path)
                    metrics['mesh'] = path.name
                reconstruction_s = perf_counter() - started
                quality = mesh_accuracy(mesh, truth)
                metrics.update(fscore_5_mm=quality['at_mm']['5']['fscore'], fscore_10_mm=quality['at_mm']['10']['fscore'],
                               precision_5_mm=quality['at_mm']['5']['precision'], recall_5_mm=quality['at_mm']['5']['recall'],
                               symmetric_mean_mm=quality['symmetric_mean_mm'], mesh_triangles=len(mesh.faces),
                               lengthscale_m=length, watertight=bool(mesh.is_watertight), reconstruction_wall_s=reconstruction_s)
                status = 'mesh'
        else:
            status = 'insufficient_voxels'
        scores.append(dict(metrics, status=status))
        print(f'{arm}: signed GPIS at {source["time_s"]:.2f}s, F@5 {metrics["fscore_5_mm"]:.2%}', flush=True)
    final = scores[0]
    scores.sort(key=lambda e: e['time_s'])
    return dict(arm=arm, trace_sha256=run['trace_sha256'], final=final, snapshots=scores,
                first_scored_snapshot_above_fscore_5_mm_s={
                    str(threshold): next((e['time_s'] for e in scores if e['fscore_5_mm'] >= threshold / 100), None)
                    for threshold in (40, 60, 80, 90, 95, 99)},
                acquisition_plus_final_reconstruction_wall_s=(run['wall_execution_s'] + final['reconstruction_wall_s']
                                                                if 'reconstruction_wall_s' in final else None))


def draw(root, runs, experiment):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig = plt.figure(figsize=(11.8, 7.), facecolor='#f4f7f5', layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1., 1.2])
    fig.suptitle(f'{experiment} / Signed GPIS from tactile measurements', fontweight='bold', fontsize=16, x=.02, ha='left')
    ax = fig.add_subplot(grid[0, :])
    for index, run in enumerate(runs):
        curve, final = run['snapshots'], run['final']
        ax.plot([e['time_s'] for e in curve], [e['fscore_5_mm'] for e in curve], 'o-', color=COLORS[index],
                lw=2, markersize=4, label=DISPLAY_NAMES[index])
        sample = fig.add_subplot(grid[1, index], projection='3d', facecolor='#f4f7f5')
        if final['mesh']:
            mesh = trimesh.load(root / run['arm'] / final['mesh'], process=False, force='mesh')
            sample.plot_trisurf(*mesh.vertices.T, triangles=mesh.faces, color=COLORS[index], shade=True,
                                 antialiased=True, linewidth=0.)
            sample.set_box_aspect(np.ptp(mesh.bounds, axis=0))
        sample.view_init(elev=26, azim=60)
        sample.set_axis_off()
        sample.set_title(f'{DISPLAY_NAMES[index]}  |  F@5 {final["fscore_5_mm"]:.2%}', fontsize=10)
    ax.set(xlabel='Exploration time (simulated seconds)', ylabel='F-score at 5 mm', ylim=(0, 1.04))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.2)
    ax.legend(loc='lower right', frameon=False)
    fig.savefig(root / 'signed-comparison.png', dpi=170)
    plt.close(fig)


def draw_progression(root, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sequences = [[entry for entry in run['snapshots'] if entry['mesh']] for run in runs]
    columns = max(2, min(4, max(map(len, sequences))))
    fig, axes = plt.subplots(2, columns, figsize=(3.4 * columns, 6.3), facecolor='#f4f7f5', layout='constrained',
                             subplot_kw={'projection': '3d', 'facecolor': '#f4f7f5'})
    fig.suptitle('active-02-a / Pear reconstruction from measured touch', fontweight='bold', fontsize=16)
    truth, _ = evaluation_reference()
    bounds = np.vstack((truth.min(axis=0), truth.max(axis=0))) + np.array([[-.009], [.009]])
    for row, (run, meshes) in enumerate(zip(runs, sequences)):
        for ax in axes[row]:
            ax.set_axis_off()
        if not meshes:
            continue
        indices = np.rint(np.linspace(0, len(meshes) - 1, min(columns, len(meshes)))).astype(int)
        for column, index in enumerate(indices):
            snap = meshes[index]
            mesh = trimesh.load(root / run['arm'] / snap['mesh'], process=False, force='mesh')
            ax = axes[row, column]
            ax.plot_trisurf(*mesh.vertices.T, triangles=mesh.faces, color=COLORS[row],
                            antialiased=True, linewidth=0., shade=True)
            ax.set(xlim=bounds[:, 0], ylim=bounds[:, 1], zlim=bounds[:, 2])
            ax.set_box_aspect(np.ptp(bounds, axis=0))
            ax.view_init(elev=23, azim=46)
            title = f'{snap["time_s"]:.0f} s  |  C10 {snap["coverage_10_mm"]:.1%}'
            if column == 0:
                title = DISPLAY_NAMES[row] + '\n' + title
            ax.set_title(title, fontsize=10)
    fig.savefig(root / 'signed-progression.png', dpi=162)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    root = parser.parse_args().output.resolve()
    base = root / 'results.json'
    scoring = read_json(base)
    maximum_load = read_json(root / 'protocol.json')['qualified_surface_region_load_n']
    if [run['arm'] for run in scoring['runs']] != list(ARMS):
        raise ValueError('Signed GPIS comparison requires both finished arms')
    with ProcessPoolExecutor(max_workers=2) as workers:
        runs = list(workers.map(evaluate, [(root, run, maximum_load) for run in scoring['runs']]))
    draw(root, runs, scoring['experiment'])
    if scoring['experiment'] == 'active-02-a':
        draw_progression(root, runs)
    write_json(root / 'signed-results.json', dict(experiment=scoring['experiment'], source_sha256=digest(Path(__file__)),
                                                coverage_results_sha256=digest(base), kernel='Matérn 3/2 (GPyTorch)',
                                                signed_sample_offset_m=NORMAL_OFFSET_M, diagonal_noise=NOISE_VARIANCE,
                                                support_radius_m=rgpis.CONTACT_SUPPORT_M, grid_limit_m=rgpis.GRID_STEP_M,
                                                contact_budget=rgpis.CONTACT_BUDGET, voxel_m=rgpis.VOXEL_M,
                                                maximum_region_load_n=maximum_load, runs=runs))


if __name__ == '__main__':
    main()
