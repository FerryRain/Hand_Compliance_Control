"""Prepare and run the paired i01-a wrist and finger-control experiments."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.transform import Rotation, RotationSpline
import trimesh

from .fingertip_benchmark import digest, read
from .fingertip_hand import DT, ROOT, FingertipHand, build_model, reference_grasp
from .fingertip_plan import JointPatchPlanner
from .fingertip_replay import chunks
from .fingertip_traversal import write_json
from .fixtures import SurfaceGraph
from .i01_a_contact_wrist import fit_contact_guide
from .i01_a_full_wrist import GUIDE_STEP_M, MAX_SCAN_SPEED_M_S, RigidFootprint, routing_adjacency
from .i01_a_wrist import STARTUP_S, SETTLE_S, as_wxyz, time_parameterize, wrist_digest
from .oracle import align_vector


GENERATED = ROOT / 'Module/generated'
DEFAULT_OUTPUT = GENERATED / 'i01_a_fair_20261121'
SPEED_CALIBRATION = GENERATED / 'i01_a_20261120/wrist_calibration.json'
DEVELOPMENT = GENERATED / 'i01_a_multicontact_20261120'
METHODS = ('fixed-dp', 'fixed-mcc', 'fixed-grasp')
OBJECTS = ('meat-can', 'ellipsoid')
STARTS = (('start-01', np.array([0., 0., 1.])), ('start-02', np.array([0., 0., -1.])))
REGIMES = ('shared', 'optimized')
SAMPLES_PER_OBJECT = 256
FOOTPRINT_WEIGHT = .75
SUPPORTS = 3
PLANNED_NEIGHBORHOOD_M = .025
ELLIPSOID_RADII_M = (.105, .137, .093)
MEAT_CAN_ASSET = (ROOT.parent / 'ETac/BlindGrasping/data/meshdata/urdf/ycb/'
                  '010_potted_meat_can/collision.obj')


def calibrate_hand_shapes():
    """Use the same number of qualified frames from each matched development run."""
    offsets, sources = {}, {}
    for method in METHODS:
        corrections, receipts = [], []
        for obj in ('mustard', 'cracker'):
            case = DEVELOPMENT / obj / f'{method}_r01'
            plan = read(case / 'wrist_plan.json')
            trace = list(chunks(case))
            joints = np.concatenate([block['qpos'][:, 7:] for block in trace])
            with np.load(case / 'moving_contact.npz', allow_pickle=False) as data:
                qualified = data['moving'] & (data['head_count'] >= 2) & (data['region_load_n'] < 8.)
            eligible = np.flatnonzero(qualified)
            if len(eligible) < SAMPLES_PER_OBJECT or joints.shape != (len(qualified), 16):
                raise ValueError(f'Insufficient matched calibration frames in {case}')
            positions = np.rint(np.linspace(0, len(eligible) - 1, SAMPLES_PER_OBJECT)).astype(int)
            sample = eligible[positions]
            corrections.append(np.median(joints[sample] - plan['seed_joints_rad'], axis=0))
            receipts.append(dict(case=case.relative_to(ROOT).as_posix(),
                                 trace_sha256=digest(case / 'trace_chunks.json'),
                                 measured_contact_sha256=digest(case / 'moving_contact.npz'),
                                 frames=sample.tolist()))
        offsets[method] = np.mean(corrections, axis=0).tolist()
        sources[method] = receipts
    return offsets, sources


def object_fixture(name, output):
    output.mkdir(parents=True)
    if name == 'meat-can':
        solid = trimesh.load_mesh(MEAT_CAN_ASSET, process=True).convex_hull
        solid.apply_translation(-solid.bounds.mean(axis=0))
        # The asset's URDF uses centimeters. Match the existing YCB experiments' 2.8 scale.
        solid.apply_scale(.01 * 2.8)
        surface = solid.copy()
        for _ in range(4):
            surface = surface.subdivide()
        source = dict(kind='YCB_010_convex_collision_shape', path=MEAT_CAN_ASSET.relative_to(ROOT.parent).as_posix(),
                      source_sha256=digest(MEAT_CAN_ASSET), scale_from_native=.028)
    else:
        surface = trimesh.creation.icosphere(subdivisions=4, radius=1.)
        surface.apply_scale(ELLIPSOID_RADII_M)
        solid = surface.copy()
        source = dict(kind='procedural_ellipsoid', radii_m=list(ELLIPSOID_RADII_M))
    if not surface.is_watertight:
        raise ValueError(f'Non-watertight reference surface for {name}')
    surface_path, solid_path = output / 'surface.obj', output / 'collision.obj'
    surface.export(surface_path)
    solid.export(solid_path)
    return surface, surface_path, solid_path, source


def initial_grasp(hand, graph, facing):
    """Find a common low-load initial grasp using only the scene geometry."""
    planner = JointPatchPlanner(hand, graph)
    center = graph.mesh.bounds.mean(axis=0)
    delta = graph.vertices_m - center
    depth = delta @ facing
    lateral = np.linalg.norm(delta - depth[:, None] * facing, axis=1)
    merit = lateral + .50 * (depth.max() - depth)
    candidates = np.flatnonzero((graph.normals @ facing) >= .85)
    ordered = candidates[np.argsort(merit[candidates], kind='stable')]
    base = align_vector(np.array([0., 0., 1.]), facing) * reference_grasp()[1]
    vertices = []
    for vertex in ordered:
        if all(np.linalg.norm(graph.vertices_m[vertex] - graph.vertices_m[j]) >= .012 for j in vertices):
            vertices.append(vertex)
        if len(vertices) >= 32:
            break
    for vertex in vertices:
        for yaw in (0., np.pi / 2., -np.pi / 2., np.pi):
            orientation = Rotation.from_rotvec(facing * yaw) * base
            for anchor in (1, 0, 2, 3):
                patch = planner.fit(int(vertex), anchor, orientation)
                if patch is None:
                    continue
                posture = patch.posture
                grasp = (posture.position, posture.orientation, posture.joints)
                touch = hand.measure(hand.initialize(*grasp))
                if touch.measured_heads.all() and touch.loads.max() < 8.:
                    return grasp, touch, dict(method='shared_geometry_only_start', vertex=int(vertex),
                                              facing=facing.tolist(), yaw_rad=yaw, anchor=anchor,
                                              solver_steps=planner.solver_steps)
    raise ValueError(f'No four-tip, low-load initial grasp for facing {facing}')


def surface_guide(graph, adjacency, initial_point):
    """Connect a geodesic cover of the surface by a deterministic short open tour."""
    start = graph.nearest_vertex(initial_point)[0]
    vertices = np.r_[start, graph.required_vertices[graph.required_vertices != start]]
    distances, parents = dijkstra(adjacency, directed=False, indices=vertices, return_predecessors=True)
    direct = distances[:, vertices]
    remaining = set(range(1, len(vertices)))
    order = [0]
    while remaining:
        target = min(remaining, key=lambda j: (direct[order[-1], j], j))
        order.append(target)
        remaining.remove(target)
    order = np.asarray(order)
    # Two-opt never moves the initial contact and includes the open end in the candidates.
    for _ in range(40):
        a, b = np.triu_indices(len(order) - 1, k=1)
        old = direct[order[a], order[a + 1]]
        change = direct[order[a], order[b + 1]] - old
        internal = b + 2 < len(order)
        j, k = a[internal], b[internal]
        change[internal] += (direct[order[j + 1], order[k + 2]] - direct[order[k + 1], order[k + 2]])
        cut = int(np.argmin(change))
        if change[cut] > -1.e-7:
            break
        order[a[cut] + 1:b[cut] + 2] = order[a[cut] + 1:b[cut] + 2][::-1]
    segments = [np.asarray([graph.vertices_m[start]])]
    for a, b in zip(order[:-1], order[1:]):
        reverse = [vertices[b]]
        while reverse[-1] != vertices[a]:
            predecessor = int(parents[a, reverse[-1]])
            if predecessor < 0:
                raise ValueError('No mesh path between evaluation landmarks')
            reverse.append(predecessor)
        points = graph.vertices_m[reverse[::-1]]
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        time = np.linspace(0., arc[-1], int(np.ceil(arc[-1] / GUIDE_STEP_M)) + 1)[1:]
        segments.append(np.column_stack([np.interp(time, arc, coordinate) for coordinate in points.T]))
    return np.vstack(segments), len(order)


def save_model(hand):
    raw = np.empty(mujoco.mj_sizeModel(hand.model), dtype=np.uint8)
    mujoco.mj_saveModel(hand.model, None, raw)
    return raw.tobytes()


def prepare_scenario(root, obj, tag, facing, model, graph, surface, source, offsets, *, guide_builder=surface_guide):
    hand = FingertipHand(model, object_name=obj)
    if not np.array_equal(hand.q, np.arange(7, 23)):
        raise ValueError('Development and test must share the native joint ordering')
    grasp, initial, seed_info = initial_grasp(hand, graph, facing)
    hand.inward_in_palm[:] = (0., 0., -1.)
    adjacency, bridges = routing_adjacency(graph)
    guide, goal_count = guide_builder(graph, adjacency, initial.points.mean(axis=0))
    reference = RigidFootprint(hand, grasp)

    files = {}
    for key in ('shared', *offsets):
        started = perf_counter()
        joints = (grasp[2] if key == 'shared' else np.clip(
            grasp[2] + FOOTPRINT_WEIGHT * np.asarray(offsets[key]), hand.limits[:, 0], hand.limits[:, 1]))
        active = RigidFootprint(hand, (grasp[0], grasp[1], joints))
        xyz, rots, nodes, coverage, counts, info = fit_contact_guide(
            graph, adjacency, reference, grasp, initial, guide, GUIDE_STEP_M,
            supports=SUPPORTS, operating_footprint=active, startup_supports=SUPPORTS)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))]
        pose = PchipInterpolator(arc, xyz)
        attitude = RotationSpline(arc, rots)
        _, speed = time_parameterize(arc, pose, attitude, read(SPEED_CALIBRATION), max_speed=MAX_SCAN_SPEED_M_S)
        info.update(geometry_planning_s=perf_counter() - started, operating_footprint_joints_rad=joints.tolist(),
                    operating_footprint_weight=FOOTPRINT_WEIGHT if key != 'shared' else 0.,
                    startup_supports=SUPPORTS, maximum_fit_iterations_per_knot=7)
        files[key] = (pose, attitude, nodes, arc, counts, coverage, speed, info)

    nodes, arc = files['shared'][2:4]
    for value in files.values():
        np.testing.assert_array_equal(value[2], nodes)
        np.testing.assert_array_equal(value[3], arc)
    speed = np.min([value[6] for value in files.values()], axis=0)
    times = np.r_[0., np.cumsum(2. * np.diff(arc) / (speed[1:] + speed[:-1]))]
    end = int(np.ceil((STARTUP_S + times[-1]) / DT))
    at = np.clip(np.arange(end + round(SETTLE_S / DT) + 1) * DT - STARTUP_S, 0., times[-1])
    progress = PchipInterpolator(times, arc)(at)
    progress[at == times[-1]] = arc[-1]
    blob = save_model(hand)
    for key, (pose, attitude, nodes, arc, counts, coverage, _, info) in files.items():
        target = root / 'plans' / obj / tag / key
        target.mkdir(parents=True)
        poses = np.column_stack((pose(progress), as_wxyz(attitude(progress))))
        poses[at == 0.] = np.r_[grasp[0], as_wxyz(grasp[1])]
        np.savez_compressed(target / 'wrist.npz', palm_command=poses, arc_m=progress)
        np.savez_compressed(target / 'planned_surface.npz', guide_m=nodes, arc_m=arc,
                            predicted_fingertips=counts, proxy_covered=coverage.visited_vertices)
        (target / 'model.mjb').write_bytes(blob)
        (target / 'surface.obj').write_bytes(surface.read_bytes())
        receipt = dict(object_name=obj, start_tag=tag, source={**source, **seed_info},
                       seed_pose_wxyz=np.r_[grasp[0], as_wxyz(grasp[1])].tolist(), seed_joints_rad=grasp[2].tolist(),
                       initial_head_forces_n=initial.forces.tolist(), initial_region_load_n=initial.loads.tolist(),
                       model_sha256=sha256(blob).hexdigest(), surface_sha256=digest(surface),
                       calibration_sha256=digest(SPEED_CALIBRATION), wrist_planner='common_three_tip_geometry',
                       footprint_owner=key, guide_length_m=float(arc[-1]), path_length_m=float(arc[-1]),
                       planning_knots=len(arc), surface_legs=goal_count - 1, component_bridges=bridges,
                       routing_landmark_cover_radius_m=graph.realized_cover_radius_m,
                       nominal_23mm_proxy_area=coverage.area_fraction, scan_end_step=end, scan_end_s=end * DT,
                       nominal_total_s=len(poses)*DT, scan_time_s=float(times[-1]),
                       nominal_wrist_steps=len(poses), nominal_wrist_sha256=wrist_digest(poses),
                       common_progress_sha256=sha256(np.ascontiguousarray(progress, dtype='<f8').tobytes()).hexdigest(),
                       physics_rollouts_in_planner=0, finger_ik_in_planner=0, **info)
        write_json(target / 'wrist_plan.json', receipt)
    print(f'{obj}/{tag}: {arc[-1]:.3f} m, {len(progress)*DT:.1f} simulated s, '
          f'initial forces {initial.forces.round(2)} N', flush=True)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    offsets, sources = calibrate_hand_shapes()
    write_json(root / 'protocol.json', dict(
        experiment='i01-a fair paired comparison', objects=OBJECTS, starts=[key for key, _ in STARTS],
        methods=METHODS, regimes=REGIMES,
        experiments={'E1': ['start-01/shared'], 'E2': ['start-01/optimized'],
                     'E3': ['start-02/shared', 'start-02/optimized']},
        planned_motion_equal_within_start=True, finger_threshold_n=.2, low_region_load_n=8.,
        overforce_region_load_n=40., motion_dt_s=DT, primary='moving_at_least_two_heads',
        prespecified_additional='moving_three_heads_below_8n', coverage_radii_mm=[10, 25],
        three_tip_geometry_target=SUPPORTS, startup_supports=SUPPORTS, maximum_fit_iterations_per_knot=7,
        route_landmark_cover_radius_m=PLANNED_NEIGHBORHOOD_M, development_samples_per_object=SAMPLES_PER_OBJECT,
        development_shape_weight=FOOTPRINT_WEIGHT, joint_offsets_rad=offsets, development_sources=sources,
        seeds={'start-01': 20261121, 'start-02': 20261122}, frozen_before_test_rollouts=True))
    for obj in OBJECTS:
        mesh, surface, solid, source = object_fixture(obj, root / 'objects' / obj)
        graph = SurfaceGraph(mesh, coverage_radius_m=PLANNED_NEIGHBORHOOD_M)
        model = build_model(surface=surface, collision_parts=[solid], object_name=obj, model_name=f'i01-a-fair-{obj}')
        for tag, facing in STARTS:
            prepare_scenario(root, obj, tag, facing, model, graph, surface, source, offsets)


def locations(root, obj, tag, regime, method):
    plan = root / 'plans' / obj / tag / ('shared' if regime == 'shared' else method)
    run = root / 'runs' / obj / tag / regime
    if regime == 'optimized':
        run /= method
    return plan, run, run / f'{method}_r01'


def execute(root, obj, tag, regime, method, seed):
    plan, run, case = locations(root, obj, tag, regime, method)
    if (case / 'stopwatch.json').is_file():
        return
    command = [sys.executable, '-u', '-m', 'Module.i04_geometric_traversal.i01_a', 'suite',
               '--root', str(run), '--prepared', str(plan), '--branches', method, '--repeats', '1',
               '--seed', str(seed), '--calibration', str(SPEED_CALIBRATION)]
    run.mkdir(parents=True, exist_ok=True)
    print(f'Start {obj}/{tag}/{regime}/{method}', flush=True)
    with (run / f'{method}_suite.log').open('wb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'Completed {obj}/{tag}/{regime}/{method}', flush=True)


def run(root, jobs, dp_jobs, only):
    protocol = read(root / 'protocol.json')
    expected = evaluation_tasks(only)
    for obj, tag, regime, method in expected:
        plan, _, _ = locations(root, obj, tag, regime, method)
        if not (plan / 'wrist_plan.json').is_file():
            raise ValueError(f'Prepare all declared wrist plans before running the tests: {plan}')
    with ThreadPoolExecutor(max_workers=jobs) as cpu, ThreadPoolExecutor(max_workers=dp_jobs) as gpu:
        tasks = []
        for obj, tag, regime, method in expected:
            pool = gpu if method == 'fixed-dp' else cpu
            tasks.append(pool.submit(execute, root, obj, tag, regime, method, protocol['seeds'][tag]))
        for future in as_completed(tasks):
            future.result()


def evaluation_tasks(only=None):
    return [(obj, tag, regime, method)
            for tag, _ in STARTS for regime in REGIMES for obj in OBJECTS for method in METHODS
            if only is None or obj == only]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'all'))
    parser.add_argument('--root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--cpu-jobs', type=int, default=4)
    parser.add_argument('--dp-jobs', type=int, default=2)
    parser.add_argument('--only', choices=OBJECTS)
    args = parser.parse_args()
    if args.action in ('prepare', 'all'):
        prepare(args.root.resolve())
    if args.action in ('run', 'all'):
        run(args.root.resolve(), args.cpu_jobs, args.dp_jobs, args.only)


if __name__ == '__main__':
    main()
