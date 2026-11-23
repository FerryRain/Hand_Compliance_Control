"""Prepare a common, continuous wrist-only coverage route for each i01-a object."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, RotationSpline
import trimesh

from .contracts import ExperimentConfig
from .fingertip_hand import DT, FIXTURE, ROOT, SEED, SURFACE, FingertipHand, build_model, reference_grasp
from .fingertip_traversal import write_json
from .fixtures import SurfaceGraph
from .i01_a_wrist import STARTUP_S, SETTLE_S, WristTape, as_wxyz, time_parameterize, wrist_digest
from .oracle import MeasuredCoverage, align_vector
from .physics import rotation


REVIEW = ROOT / 'mcc_finger_compliance_control/data/contact_wm_review_20261110'
EVALUATION = (ROOT / 'finger_policy/data/evaluation_runs/single_all_generalization/evaluation'
              / 'single_all_seed20261116')
ASSETS = {
    'mustard': (SURFACE, FIXTURE / 'collision_parts', None, 0),
    'pear': (REVIEW / 'ycb_expansion_20261113/objects/ycb_016_pear/surface.obj',
             REVIEW / 'physics_contract_v1/controlled_policy_ablation_20261116/fruit_generalization/ycb_016_pear/collision',
             EVALUATION / 'ycb_016_pear_forward.npz', 500),
    'cracker': (REVIEW / 'ycb_expansion_v2/objects/ycb_003_cracker_box/surface.obj',
                REVIEW / 'physics_contract_v1/ycb_policy_stress_v1/ycb_003_cracker_box/collision',
                EVALUATION / 'ycb_003_cracker_box.npz', 0),
}
GUIDE_STEP_M = .008
MAX_SCAN_SPEED_M_S = .031
KNOT_DEPTH_M = .00085
FOOTPRINT_GOAL = .997


def unit(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1.e-12)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def fixture(name):
    surface, collision_dir, source, frame = ASSETS[name]
    if source is None:
        grasp = reference_grasp()
        source = SEED
        files = sorted(collision_dir.glob('part_*.obj'))
    else:
        with np.load(source, allow_pickle=False) as logged:
            if not logged['observed'][frame, 0]:
                raise ValueError(f'{name} initialization frame was not observed')
            object_pose = logged['measured_object_pose_world'][frame, 0].astype(float)
            palm = logged['input_palm_pose_world'][frame, 0].astype(float)
            object_rotation = rotation(object_pose[3:])
            grasp = (object_rotation.inv().apply(palm[:3] - object_pose[:3]),
                     object_rotation.inv() * rotation(palm[3:]), logged['input_q_live'][frame, 0].astype(float))
        files = sorted(collision_dir.glob('collision_part_*.obj'))
    if len(files) != (256 if name == 'mustard' else 128):
        raise ValueError(f'Incomplete collision asset for {name}')
    model = build_model(surface=surface, collision_parts=files, object_name=name, model_name=f'i01-a-{name}')
    hand = FingertipHand(model, object_name=name, grasp=grasp)
    hand.inward_in_palm[:] = (0., 0., -1.)
    return surface, hand, grasp, dict(source=source.relative_to(ROOT).as_posix(), sha256=digest(source), frame=frame)


def routing_adjacency(graph):
    adjacency = graph.adjacency
    components, labels = connected_components(adjacency, directed=False)
    if components == 1:
        return adjacency, []
    adjacency = adjacency.tolil(copy=True)
    connected = labels == labels[0]
    bridges = []
    for _ in range(components - 1):
        joined, pending = np.flatnonzero(connected), np.flatnonzero(~connected)
        distance, nearest = cKDTree(graph.vertices_m[joined]).query(graph.vertices_m[pending])
        i = int(distance.argmin())
        a, b = int(joined[nearest[i]]), int(pending[i])
        adjacency[a, b] = adjacency[b, a] = float(max(distance[i], 1.e-8))
        bridges.append([a, b, float(distance[i])])
        connected |= labels == labels[b]
    return adjacency.tocsr(), bridges


class RigidFootprint:
    """Fit a fixed hand footprint by translating the six-DoF wrist along the surface normal."""

    def __init__(self, hand, grasp):
        self.h = hand
        position, attitude, joints = grasp
        self.data = hand.initialize(position, attitude, joints)
        self.collision_queries = 0
        self.heads = self.support_points(position, attitude)
        # The lowest distal support is used to locate the wrist on a flat face.
        self.anchor = int(self.heads[:, 2].argmin())

    def support_points(self, position, attitude):
        model, data = self.h.model, self.data
        points = []
        for f, geom in enumerate(self.h.pads):
            mesh = model.geom_dataid[geom]
            first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            world = data.geom_xpos[geom] + model.mesh_vert[first:first + count] @ data.geom_xmat[geom].reshape(3, 3).T
            front = np.array([self.h.fraction(data, f, point) >= self.h.min_head_fraction for point in world])
            local = attitude.inv().apply(world[front] - position)
            if len(local) == 0:
                raise ValueError('The distal fingertip mesh has no support vertices')
            points.append(local[local[:, 2] < local[:, 2].min() + .00015].mean(axis=0))
        return np.array(points)

    def collisions(self, position, quaternion):
        model, data = self.h.model, self.data
        data.qpos[:3], data.qpos[3:7] = position, quaternion
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_collision(model, data)
        self.collision_queries += 1
        result = []
        objects = self.h.objects
        for contact in data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if (a in objects) != (b in objects):
                outward = contact.frame[:3].copy() * (1 if a in objects else -1)
                result.append((float(contact.dist), contact.pos.copy(), outward, b if a in objects else a))
        return result

    def fit(self, target, attitude, preload=KNOT_DEPTH_M):
        normal = attitude.apply([0., 0., 1.])
        origin = target - attitude.apply(self.heads[self.anchor])
        return self.settle(origin, attitude, normal, preload)

    def settle(self, origin, attitude, normal, preload, offset=0.):
        quaternion = as_wxyz(attitude)
        for _ in range(10):
            contacts = self.collisions(origin + offset * normal, quaternion)
            nearest = min(contacts, key=lambda row: row[0], default=None)
            if nearest is None:
                change = -.003
            else:
                delta = -preload - nearest[0]
                if abs(delta) < .000075:
                    break
                change = np.clip(delta / max(.3, nearest[2] @ normal), -.006, .009)
            new = np.clip(offset + change, -.027, .065)
            if abs(new - offset) < 1.e-7:
                break
            offset = new
        position = origin + offset * normal
        contacts = self.collisions(position, quaternion)
        return position, *self.head_contacts(contacts)

    def head_contacts(self, contacts):
        points, normals = np.zeros((4, 3)), np.zeros((4, 3))
        distances = np.full(4, np.inf)
        for distance, point, outward, geom in contacts:
            f = self.h.pad_index.get(geom)
            if (f is not None and distance < distances[f]
                    and self.h.fraction(self.data, f, point) >= self.h.min_head_fraction):
                points[f], normals[f], distances[f] = point, outward, distance
        return points, normals, distances <= .00025


def next_attitude(attitude, normal):
    change = align_vector(attitude.apply([0., 0., 1.]), normal).as_rotvec()
    angle = np.linalg.norm(change)
    return Rotation.from_rotvec(change * min(1., .15 / max(angle, 1.e-12))) * attitude


def route_nodes(graph, hand, grasp, initial):
    adjacency, bridges = routing_adjacency(graph)
    distances, neighbors = graph._tree.query(graph.vertices_m, k=24)
    weights = np.exp(-(distances / .018)**2)
    smooth_normals = unit(np.einsum('nk,nkj->nj', weights, graph.normals[neighbors]))
    fit = RigidFootprint(hand, grasp)
    start_force = initial.forces[fit.anchor]
    if start_force >= .2:
        start = graph.nearest_vertex(initial.points[fit.anchor])[0]
    else:
        start = graph.nearest_vertex(initial.points[initial.measured_heads].mean(axis=0))[0]
    current = int(start)
    position, attitude, _ = grasp
    xyz, rots, guide = [position], [attitude], [graph.vertices_m[current]]
    predicted_count = [int(initial.measured_heads.sum())]
    proxy = MeasuredCoverage(graph, ExperimentConfig(coverage_radius_m=.023))
    intended = np.zeros(len(graph.vertices_m), bool)
    intended[proxy.neighborhood(current)] = True
    proxy.update(initial.points, initial.normals, initial.measured_heads)
    attempted = np.zeros(len(intended), bool)
    goals = graph.required_vertices
    chosen = []
    # Prefer a new reachable surface neighborhood. Keep routing even when a
    # tight corner cannot produce a rigid distal footprint at a guide knot.
    extra = 0
    while True:
        missing = ~(proxy.visited_vertices | intended) & ~attempted
        remaining = goals[missing[goals]]
        if not len(remaining):
            remaining = np.flatnonzero(missing)
        if not len(remaining) and proxy.area_fraction < FOOTPRINT_GOAL and extra < 36:
            remaining = goals[~proxy.visited_vertices[goals] & ~attempted[goals]]
            extra += 1
        if not len(remaining):
            break
        distance, previous = dijkstra(adjacency, directed=False, indices=current, return_predecessors=True)
        geometric_cost = distance[remaining] + .028 * (1. - smooth_normals[remaining] @ smooth_normals[current])
        goal = int(remaining[geometric_cost.argmin()])
        attempted[goal] = True
        vertices = [goal]
        while vertices[-1] != current:
            vertices.append(int(previous[vertices[-1]]))
            if vertices[-1] < 0:
                raise ValueError('Unreachable surface route')
        vertices = vertices[::-1]
        if len(vertices) < 2:
            continue
        line = graph.vertices_m[vertices]
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
        stations = np.linspace(0., arc[-1], max(2, int(np.ceil(arc[-1] / GUIDE_STEP_M)) + 1))[1:]
        points = np.column_stack([np.interp(stations, arc, axis) for axis in line.T])
        normals = unit(np.column_stack([np.interp(stations, arc, axis) for axis in smooth_normals[vertices].T]))
        for point, normal in zip(points, normals):
            if np.linalg.norm(point - guide[-1]) < 1.e-7:
                continue
            attitude = next_attitude(attitude, normal)
            position, tip, tip_normals, valid = fit.fit(point, attitude)
            if not np.isfinite(position).all():
                raise ValueError('Nonfinite wrist pose')
            xyz.append(position)
            rots.append(attitude)
            guide.append(point)
            predicted_count.append(int(valid.sum()))
            proxy.update(tip, tip_normals, valid)
        for vertex in vertices:
            if not intended[vertex]:
                intended[proxy.neighborhood(vertex)] = True
        current = goal
        chosen.append(goal)
        if len(chosen) % 45 == 0:
            print(f'  {len(chosen)} surface legs, proxy coverage {proxy.area_fraction:.1%}', flush=True)
    return (np.asarray(xyz), Rotation.concatenate(rots), np.asarray(guide), proxy, intended, chosen,
            np.asarray(predicted_count), bridges, fit.collision_queries)


def prepare(name, output, calibration, wrist_planner='policy-footprint'):
    started = perf_counter()
    surface, hand, grasp, source = fixture(name)
    mesh = trimesh.load_mesh(surface, process=False)
    graph = SurfaceGraph(mesh, coverage_radius_m=.025)
    initial = hand.measure(hand.initialize(*grasp))
    if not initial.measured_heads.all() or initial.loads.max() >= 8.:
        raise ValueError(f'{name} does not begin at the qualified four-fingertip grasp: {initial.forces}')
    planning = perf_counter()
    xyz, rots, guide, proxy, intended, goals, count, bridges, queries = route_nodes(graph, hand, grasp, initial)
    planner_info = dict(wrist_planner=wrist_planner, variables_optimized_per_knot=1)
    if wrist_planner in ('multi-contact', 'policy-footprint'):
        from .i01_a_contact_wrist import POLICY_REFERENCE_WEIGHT, fit_contact_guide, policy_reference

        fit = RigidFootprint(hand, grasp)
        operating = None
        if wrist_planner == 'policy-footprint':
            typical_joints, sources = policy_reference(EVALUATION)
            probe_joints = (1. - POLICY_REFERENCE_WEIGHT) * grasp[2] + POLICY_REFERENCE_WEIGHT * typical_joints
            operating = RigidFootprint(hand, (grasp[0], grasp[1], probe_joints))
            planner_info.update(operating_footprint_joints_rad=probe_joints.tolist(), footprint_sources=sources)
        adjacency, _ = routing_adjacency(graph)
        xyz, rots, guide, proxy, count, metrics = fit_contact_guide(
            graph, adjacency, fit, grasp, initial, guide, GUIDE_STEP_M,
            supports=3 if operating is not None else 2, operating_footprint=operating)
        planner_info.update(metrics)
        queries += fit.collision_queries + (operating.collision_queries if operating is not None else 0)
    elif wrist_planner != 'single-anchor':
        raise ValueError(f'Unknown wrist planner {wrist_planner}')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(guide, axis=0), axis=1))]
    pos, ori = PchipInterpolator(arc, xyz), RotationSpline(arc, rots)
    times, speeds = time_parameterize(arc, pos, ori, json.loads(calibration.read_bytes()), max_speed=MAX_SCAN_SPEED_M_S)
    end = int(np.ceil((STARTUP_S + times[-1]) / DT))
    at = np.clip(np.arange(end + round(SETTLE_S / DT) + 1) * DT - STARTUP_S, 0., times[-1])
    progress = PchipInterpolator(times, arc)(at)
    progress[at == times[-1]] = arc[-1]
    poses = np.column_stack((pos(progress), as_wxyz(ori(progress))))
    poses[at == 0.] = np.r_[grasp[0], as_wxyz(grasp[1])]
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'wrist.npz', palm_command=poses, arc_m=progress)
    np.savez_compressed(output / 'planned_surface.npz', guide_m=guide, arc_m=arc,
                        predicted_fingertips=count, proxy_covered=proxy.visited_vertices, intended_covered=intended,
                        goals=np.asarray(goals))
    (output / 'surface.obj').write_bytes(surface.read_bytes())
    blob = np.empty(mujoco.mj_sizeModel(hand.model), dtype=np.uint8)
    mujoco.mj_saveModel(hand.model, None, blob)
    (output / 'model.mjb').write_bytes(blob.tobytes())
    report = dict(object_name=name, source=source, seed_pose_wxyz=np.r_[grasp[0], as_wxyz(grasp[1])].tolist(),
                  seed_joints_rad=grasp[2].tolist(), initial_head_forces_n=initial.forces.tolist(),
                  surface_sha256=digest(output / 'surface.obj'), model_sha256=sha256(blob).hexdigest(),
                  calibration_sha256=digest(calibration), mesh_faces=len(graph.faces), mesh_area_m2=float(mesh.area),
                  guide_length_m=float(arc[-1]), path_length_m=float(arc[-1]),
                  wrist_path_length_m=float(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1).sum()),
                  planning_knots=len(arc), surface_legs=len(goals), positional_collision_queries=queries,
                  component_bridges=bridges, intended_23mm_area=float(proxy.vertex_area @ intended / proxy.vertex_area.sum()),
                  combined_23mm_area=float(proxy.vertex_area @ (intended | proxy.visited_vertices) / proxy.vertex_area.sum()),
                  nominal_23mm_proxy_area=proxy.area_fraction,
                  fraction_knots_with_proxy_head=float(np.mean(count > 0)),
                  fraction_knots_with_two_proxy_heads=float(np.mean(count >= 2)),
                  fraction_knots_with_three_proxy_heads=float(np.mean(count >= 3)),
                  scan_end_step=end, scan_end_s=end * DT, nominal_total_s=len(poses) * DT,
                  scan_time_s=float(times[-1]), maximum_scan_speed_m_s=float(speeds.max()),
                  nominal_wrist_steps=len(poses), nominal_wrist_sha256=wrist_digest(poses),
                  geometry_planning_s=perf_counter() - planning,
                  preparation_s=perf_counter() - started, physics_rollouts_in_planner=0, finger_ik_in_planner=0,
                  **planner_info)
    write_json(output / 'wrist_plan.json', report)
    print(f'{name}: {arc[-1]:.3f} m, {len(poses)*DT:.1f} s simulated, '
          f'intended {report["intended_23mm_area"]:.1%}, distal proxy {proxy.area_fraction:.1%}, '
          f'planned in {report["preparation_s"]:.2f} s', flush=True)


@dataclass
class PreparedWrist:
    surface: Path
    hand: FingertipHand
    grasp: tuple
    graph: SurfaceGraph
    tape: WristTape
    report: dict


def load_prepared(path):
    report = json.loads((path / 'wrist_plan.json').read_bytes())
    surface, model_file = path / 'surface.obj', path / 'model.mjb'
    if digest(surface) != report['surface_sha256'] or digest(model_file) != report['model_sha256']:
        raise ValueError('Prepared scene differs from the wrist plan')
    name = report['object_name']
    palm = np.asarray(report['seed_pose_wxyz'])
    grasp = (palm[:3], rotation(palm[3:]), np.asarray(report['seed_joints_rad']))
    model = mujoco.MjModel.from_binary_path('scene.mjb', {'scene.mjb': model_file.read_bytes()})
    hand = FingertipHand(model, object_name=name, grasp=grasp)
    hand.inward_in_palm[:] = (0., 0., -1.)
    with np.load(path / 'wrist.npz', allow_pickle=False) as tape:
        wrist, arc = tape['palm_command'], tape['arc_m']
    if wrist_digest(wrist) != report['nominal_wrist_sha256'] or len(wrist) != report['nominal_wrist_steps']:
        raise ValueError('Prepared wrist commands differ from the shared reference')
    return PreparedWrist(surface, hand, grasp, SurfaceGraph(trimesh.load_mesh(surface, process=False)),
                        WristTape(wrist, arc, report['scan_end_step'], np.zeros(3), np.zeros((4, 3)), report), report)
