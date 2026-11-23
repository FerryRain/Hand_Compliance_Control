"""Run three finger controllers over locked whole-surface spirals on new curved shapes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra
import trimesh

from .fingertip_benchmark import digest, read
from .fingertip_hand import build_model
from .fingertip_traversal import write_json
from .fixtures import SurfaceGraph
from .i01_a_fair import GENERATED, METHODS, SPEED_CALIBRATION, execute, locations, prepare_scenario
from .i01_a_full_wrist import GUIDE_STEP_M


DEFAULT_OUTPUT = GENERATED / 'i01_a_smooth_20261121'
SHAPES = {
    'oblate': dict(name='扁椭球', axes_m=(.117, .154, .096), exponent=2., taper=0.),
    'ovoid': dict(name='不对称卵形体', axes_m=(.110, .130, .150), exponent=2., taper=.15),
    'superellipsoid': dict(name='超椭球', axes_m=(.118, .139, .103), exponent=2.7, taper=0.),
}
STARTS = {
    'start-01': dict(name='起始抓姿 A', facing=(0., 0., 1.), seed=20261124),
    'start-02': dict(name='起始抓姿 B', facing=(0., 0., -1.), seed=20261125),
    'start-03': dict(name='起始抓姿 C', facing=(0., 1., 0.), seed=20261126),
}
COVER_RADIUS_M = .025
MIN_SUSTAINED_S = 1.


def surface_points(directions, shape):
    radius = (np.abs(directions)**shape['exponent']).sum(axis=1)**(-1. / shape['exponent'])
    points = directions * radius[:, None]
    points[:, :2] *= 1. + shape['taper'] * points[:, 2, None]
    return points * shape['axes_m']


def spiral_guide(graph, adjacency, start, *, shape, summary):
    pole = start / shape['axes_m']
    pole[:2] /= 1. + shape['taper'] * pole[2]
    pole /= np.linalg.norm(pole)
    axis = np.eye(3)[np.argmin(np.abs(pole))]
    tangent = axis - (axis @ pole) * pole
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(pole, tangent)
    theta = np.linspace(0., np.pi, 12289)
    lowest_turns = max(5, int(np.ceil(np.pi * max(shape['axes_m']) / (2. * COVER_RADIUS_M))))
    for turns in range(lowest_turns, lowest_turns + 10):
        phi = 2. * turns * theta
        direction = (np.cos(theta)[:, None] * pole +
                     np.sin(theta)[:, None] * (np.cos(phi)[:, None] * tangent +
                                                np.sin(phi)[:, None] * bitangent))
        points = surface_points(direction, shape)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        stations = np.linspace(0., arc[-1], int(np.ceil(arc[-1] / GUIDE_STEP_M)) + 1)
        guide = np.column_stack([np.interp(stations, arc, coordinate) for coordinate in points.T])
        _, vertex = graph._tree.query(guide)
        nearest = dijkstra(adjacency, indices=np.unique(vertex), directed=False, min_only=True)
        if nearest.max() <= COVER_RADIUS_M:
            summary.update(spiral_turns=turns, guide_vertex_radius_m=float(nearest.max()),
                           unprefixed_guide_length_m=float(arc[-1]))
            return guide, len(guide)
    raise ValueError('The prespecified spiral search did not cover the connected mesh')


def groups():
    return [(obj, start) for start in STARTS for obj in SHAPES]


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        experiment='i01-a smooth held-out shape benchmark',
        training_overlap='three exact new procedural mesh geometries, created for evaluation',
        shapes=SHAPES, starts=STARTS, methods=METHODS,
        locked_before_rollouts=True, wrist='same prerecorded wrist command tape in every matched triplet',
        geometry_target_tips=3, nominal_guide_spacing_m=GUIDE_STEP_M, guide_vertex_radius_m=COVER_RADIUS_M,
        fingertip_force_threshold_n=.2, low_region_load_n=8., overload_region_load_n=40.,
        primary_metric='fraction of commanded scan in continuous low-load three-tip bouts',
        primary_minimum_bout_s=MIN_SUSTAINED_S, primary_window='every common scan step, excluding startup and settle',
        secondary_metrics=['two fingertips', 'low-load three fingertips', 'MCC exactly two to DP at least three',
                           'scan time over 40 N', '25 mm coverage', 'R-GPIS F-score at 5 mm'],
        pair_conditional_window='commanded scan steps with pure MCC exactly two tips and all regions below 8 N',
        wrist_speed_calibration_sha256=digest(SPEED_CALIBRATION),
        checkpoint_sha256=read(SPEED_CALIBRATION)['checkpoint_sha256'])
    write_json(root / 'protocol.json', protocol)
    fixtures = {}
    for obj, shape in SHAPES.items():
        output = root / 'objects' / obj
        output.mkdir(parents=True)
        mesh = trimesh.creation.icosphere(subdivisions=4)
        mesh.vertices = surface_points(mesh.vertices, shape)
        mesh = mesh.convex_hull
        if not mesh.is_watertight or not mesh.is_convex:
            raise ValueError(f'{obj} needs a closed convex collision surface')
        surface, solid = output / 'surface.obj', output / 'collision.obj'
        mesh.export(surface)
        mesh.export(solid)
        graph = SurfaceGraph(mesh, coverage_radius_m=COVER_RADIUS_M)
        model = build_model(surface=surface, collision_parts=[solid], object_name=obj,
                            model_name=f'i01-a-smooth-{obj}')
        fixtures[obj] = (model, graph, surface)

    locked = []
    for obj, start in groups():
        model, graph, surface = fixtures[obj]
        shape = SHAPES[obj]
        geometry = {}
        make_guide = partial(spiral_guide, shape=shape, summary=geometry)
        data = dict(kind='procedural_smooth_holdout', geometry=shape)
        prepare_scenario(root, obj, start, np.array(STARTS[start]['facing']), model, graph, surface, data,
                         offsets={}, guide_builder=make_guide)
        plan, _, _ = locations(root, obj, start, 'shared', METHODS[0])
        receipt = dict(object=obj, start=start, **geometry)
        for file in ('wrist_plan.json', 'wrist.npz', 'planned_surface.npz', 'model.mjb', 'surface.obj'):
            receipt[file] = digest(plan / file)
        locked.append(receipt)
    write_json(root / 'locked_wrist_plans.json', dict(protocol_sha256=digest(root / 'protocol.json'), plans=locked))


def verify_locked(root):
    manifest = read(root / 'locked_wrist_plans.json')
    if manifest['protocol_sha256'] != digest(root / 'protocol.json'):
        raise ValueError('Benchmark protocol changed after the trajectories were locked')
    locked = {(row['object'], row['start']): row for row in manifest['plans']}
    if set(locked) != set(groups()):
        raise ValueError('The locked benchmark does not contain all nine object/start groups')
    for (obj, start), receipt in locked.items():
        plan, _, _ = locations(root, obj, start, 'shared', METHODS[0])
        for file in ('wrist_plan.json', 'wrist.npz', 'planned_surface.npz', 'model.mjb', 'surface.obj'):
            if receipt[file] != digest(plan / file):
                raise ValueError(f'Prepared benchmark changed for {obj}/{start}/{file}')


def run(root, cpu_jobs, dp_jobs):
    verify_locked(root)
    protocol = read(root / 'protocol.json')
    task_groups = groups()
    # Create shared parents before concurrent workers enter the same Windows directory.
    for obj, start in task_groups:
        _, directory, _ = locations(root, obj, start, 'shared', METHODS[0])
        directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=cpu_jobs) as cpu, ThreadPoolExecutor(max_workers=dp_jobs) as gpu:
        pending = []
        for obj, start in task_groups:
            for method in METHODS:
                runner = gpu if method == 'fixed-dp' else cpu
                pending.append(runner.submit(execute, root, obj, start, 'shared', method,
                                             protocol['starts'][start]['seed']))
        for future in as_completed(pending):
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'all'))
    parser.add_argument('--root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--cpu-jobs', type=int, default=6)
    parser.add_argument('--dp-jobs', type=int, default=4)
    args = parser.parse_args()
    output = args.root.resolve()
    if args.action in ('prepare', 'all'):
        prepare(output)
    if args.action in ('run', 'all'):
        run(output, args.cpu_jobs, args.dp_jobs)


if __name__ == '__main__':
    main()
