"""Recompute whole-surface coverage from the saved measured contact ledger."""
from __future__ import annotations

import argparse
import ast
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.sparse.csgraph import dijkstra

from Module.fr3_leap import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from Module.fr3_leap.model import compile_model_xml
from Module.e05_physics.scene import FINGERS
from .fixtures import SurfaceGraph
from .full_review import task_mesh


def audit(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    # NpzFile reads/decompresses on every __getitem__; materialize once.
    with np.load(directory/'trace.npz') as archive:
        trace = {key: archive[key] for key in archive.files}
    root = Path(__file__).resolve().parents[2]
    for name, digest in manifest['code_hashes'].items():
        source = directory/'source_snapshot/full_surface.py' if name.endswith('/full_surface.py') else root/name
        assert sha256(source.read_bytes()).hexdigest() == digest, name
    runtime_source = (directory/'source_snapshot/full_surface.py').read_text(encoding='utf-8')
    tree = ast.parse(runtime_source)
    sweep = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_sweep')
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == 'set_joint_state' for node in ast.walk(sweep))
    saved_graph = np.load(directory/'surface_graph.npz')
    graph = SurfaceGraph(task_mesh(), coverage_radius_m=.025)
    np.testing.assert_allclose(saved_graph['vertices'], graph.vertices_m, atol=1e-12)
    np.testing.assert_array_equal(saved_graph['faces'], graph.faces)
    np.testing.assert_array_equal(saved_graph['required_vertices'], graph.required_vertices)
    vertex_area = np.zeros(len(graph.vertices_m))
    np.add.at(vertex_area, graph.faces.ravel(), np.repeat(graph.mesh.area_faces/3, 3))
    mask = np.zeros(len(vertex_area), dtype=bool)
    first_touch = np.full(len(vertex_area), np.nan)
    position = np.array(manifest['object_position_m'])
    if manifest.get('resumed_development_run'):
        parent = Path(manifest['resumed_development_run'])
        with np.load(parent/'trace.npz') as old:
            mask[:] = old['covered_vertices']
        first_touch[mask] = -1.  # Explicit inherited development evidence.
    rows, fingers = np.nonzero(trace['active'])
    assert np.all(trace['forces_n'][rows, fingers] >= .2)
    points = trace['contact_positions_world_m'][rows, fingers]-position
    normals = trace['normals_world'][rows, fingers]
    assert np.all(np.isfinite(points)) and np.all(np.isfinite(normals))
    distances, candidates = graph._tree.query(points, k=24)
    unit = normals/np.linalg.norm(normals, axis=1, keepdims=True)
    alignments = np.einsum('nkj,nj->nk', graph.normals[candidates], unit)
    eligible = distances <= distances[:, :1]+.0015
    selected_column = np.argmax(np.where(eligible, alignments, -np.inf), axis=1)
    vertices = candidates[np.arange(len(rows)), selected_column]
    residual = distances[np.arange(len(rows)), selected_column]
    alignment = np.einsum('ij,ij->i', graph.normals[vertices], normals)
    valid = (residual <= .005) & (alignment >= .5)
    @lru_cache(maxsize=8192)
    def neighborhood(vertex):
        distances = dijkstra(graph.adjacency, directed=False, indices=int(vertex), limit=.025)
        return np.flatnonzero(np.isfinite(distances))
    actual_curve = np.zeros(len(trace['time_s']))
    cursor = 0
    area = float(vertex_area@mask)
    for row in range(len(actual_curve)):
        while cursor < len(rows) and rows[cursor] == row:
            if valid[cursor]:
                touched = neighborhood(int(vertices[cursor]))
                added = touched[~mask[touched]]
                first_touch[added] = trace['time_s'][row]
                area += float(vertex_area[added].sum())
                mask[touched] = True
            cursor += 1
        actual_curve[row] = area/vertex_area.sum()
    np.testing.assert_array_equal(mask, trace['covered_vertices'])
    np.testing.assert_array_equal(mask[graph.required_vertices], trace['required_goal_visited'])
    np.testing.assert_allclose(actual_curve, trace['coverage'], atol=1e-10, rtol=0)
    assert summary['covered_vertex_count'] == int(mask.sum())
    assert summary['visited_required_goals'] == int(mask[graph.required_vertices].sum())
    assert abs(summary['final_covered_area_fraction']-float(vertex_area@mask/vertex_area.sum())) < 1e-10
    measured = bool(mask.all() and mask[graph.required_vertices].all())
    assert summary['full_surface_complete'] == (measured and summary['stop_reason'] == 'FULL_MEASURED_SURFACE')
    if manifest.get('completion_policy') and summary['full_surface_complete']:
        tail = trace['time_s'] >= trace['time_s'][-1]-.15+1e-10
        assert np.all(trace['active'][tail].any(axis=1))
        assert np.all(trace['forces_n'][tail].max(axis=1) <= 8.)
    # FK of recorded states independently confirms the saved pad site path.
    model = compile_model_xml((directory/'model.xml').read_text(encoding='utf-8'))
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
           for name in (*ARM_JOINT_NAMES, *HAND_JOINT_NAMES)]
    qpos_addresses = model.jnt_qposadr[ids]
    # The site names are listed in the physical robot model, not inferred
    # from contact targets.
    tip_names = [finger.pad_site.replace('e05_', 'fr3_') for finger in FINGERS]
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in tip_names]
    if len(site_ids) != 4 or min(site_ids) < 0:
        raise ValueError('Expected exactly four physical fingertip sites')
    data = mujoco.MjData(model)
    max_tip_error = 0.
    checked_rows = np.unique(np.linspace(0, len(actual_curve)-1, 201).astype(int))
    for row in checked_rows:
        data.qpos[qpos_addresses] = trace['q_rad'][row]
        mujoco.mj_forward(model, data)
        max_tip_error = max(max_tip_error,
            float(np.max(abs(data.site_xpos[site_ids]-trace['tip_positions_world_m'][row]))))
    assert max_tip_error < 1e-9, max_tip_error
    dt = np.diff(trace['time_s'])
    dq = abs(np.diff(trace['command_rad'], axis=0))
    speed = float(np.max(dq/np.maximum(dt[:, None], 1e-12))) if len(dt) else 0.
    assert speed <= 1.2+1e-8, speed
    events = json.loads((directory/'events.json').read_text(encoding='utf-8'))
    expected_fields = ['wrist_pose'] if manifest['cell'] == 'A-MCC' else ['wrist_pose', 'tip_positions']
    assert all(e['exposed_fields'] == expected_fields for e in events if e['kind'] == 'SURFACE_TARGET')
    assert not any(e['kind'] == 'HARDWARE_STOP' for e in events) or not summary['full_surface_complete']
    np.savez_compressed(directory/'coverage_independent.npz', covered_vertices=mask,
        first_touch_time_s=first_touch, measured_coverage=actual_curve)
    result = dict(status='PASSED', inherited_development_ledger=bool(manifest.get('resumed_development_run')),
        actual_contact_samples=int(len(rows)), accepted_contact_samples=int(valid.sum()),
        accepted_contact_anchor_vertices=int(len(np.unique(vertices[valid]))),
        vertices=int(mask.sum()), total_vertices=len(mask), required_goals=int(mask[graph.required_vertices].sum()),
        total_required_goals=len(graph.required_vertices), full_surface_complete=summary['full_surface_complete'],
        all_surface_measured=bool(mask.all()),
        area_fraction=float(vertex_area@mask/vertex_area.sum()), fk_checked_states=len(checked_rows),
        maximum_fk_tip_error_m=max_tip_error, maximum_logged_command_speed_rad_s=speed,
        continuous_from_acquisition=not bool(manifest.get('resumed_development_run')),
        runtime_source_hashes_verified=True, runtime_physical_joint_teleport_calls=0,
        coverage_radius_m=.025, dp='DEFERRED_BY_USER',
        trace_sha256=sha256((directory/'trace.npz').read_bytes()).hexdigest(),
        model_xml_sha256=sha256((directory/'model.xml').read_bytes()).hexdigest())
    (directory/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('AUDIT', json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('directory', type=Path)
    audit(parser.parse_args().directory)


if __name__ == '__main__': main()
