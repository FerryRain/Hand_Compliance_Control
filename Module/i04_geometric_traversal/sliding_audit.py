"""Independent evidence checks for native contact-first surface sliding runs.

Evidence integrity and behavioral success are deliberately separate: a trace
that faithfully records a safety stop can pass this audit without passing the
experiment. Coverage is reconstructed only from the physical sensor ledger.
"""
from __future__ import annotations

import argparse
import ast
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.sparse.csgraph import dijkstra

from Module.e05_physics.scene import FINGERS
from Module.fr3_leap import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from Module.fr3_leap.model import compile_model_xml
from .fixtures import SurfaceGraph
from .full_review import task_mesh
from .physics import contact_measurement


CONTEXT_PROTOCOLS = (
    'i04-mustard-coordinated-finger-sliding.development.v3.4-context',
    'i04-mustard-coordinated-finger-sliding.development.v3.5-boundary',
    'i04-mustard-coordinated-finger-sliding.development.v3.6-prepared',
    'i04-mustard-coordinated-finger-sliding.development.v3.7-normal',
    'i04-mustard-coordinated-finger-sliding.development.v3.8-release',
)


def _source_checks(directory, manifest):
    hashes = manifest['source_hashes']
    assert hashes, 'Missing runtime source hashes'
    for relative, expected in hashes.items():
        name = PurePosixPath(relative)
        assert not name.is_absolute() and '..' not in name.parts, relative
        source = directory/'source_snapshot'/name
        assert sha256(source.read_bytes()).hexdigest() == expected, relative
    source_name = manifest.get('runtime_entrypoint', 'Module/i04_geometric_traversal/surface_slide.py')
    assert isinstance(source_name, str) and source_name in hashes, 'Declared runtime entrypoint is not frozen'
    entrypoint = PurePosixPath(source_name)
    assert not entrypoint.is_absolute() and '..' not in entrypoint.parts
    tree = ast.parse((directory/'source_snapshot'/source_name).read_text(encoding='utf-8'))
    run = next(node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == 'run')
    setters = [node for node in ast.walk(run) if isinstance(node, ast.Call)
               and ((isinstance(node.func, ast.Name) and node.func.id == 'set_joint_state')
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == 'set_joint_state'))]
    writes = []
    for node in ast.walk(run):
        targets = (node.targets if isinstance(node, ast.Assign) else [node.target]
                   if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
        for target in targets:
            if any(isinstance(item, ast.Attribute) and item.attr == 'qpos'
                   for item in ast.walk(target)):
                writes.append(node)
    assert not setters and not writes, 'Physical run assigns joints instead of stepping dynamics'
    witness = _witness_snapshot_checks(directory, manifest)
    relief = {}
    if manifest.get('protocol') == CONTEXT_PROTOCOLS[4]:
        from .release_sliding_audit import applied_reference_relief_checks
        relief = applied_reference_relief_checks(directory, manifest)
    return dict(runtime_source_hashes_verified=True, runtime_sources_checked=len(hashes),
                audited_runtime_entrypoint=source_name,
                runtime_direct_joint_teleport_calls=len(setters), runtime_direct_qpos_writes=len(writes),
                ast_scope='run only; legitimate build/acquire initialization excluded', **witness, **relief)


def _witness_snapshot_checks(directory, manifest):
    if 'witness_snapshot' not in manifest:
        return dict(geometric_witness_snapshot_verified=False,
                    witness_snapshot_scope='Legacy run did not declare a frozen geometric cache')
    name = manifest['witness_snapshot']
    assert isinstance(name, str) and name, 'Declared witness snapshot must name a file'
    relative = PurePosixPath(name)
    assert not relative.is_absolute() and '..' not in relative.parts, name
    path = (Path(directory)/relative).resolve()
    assert path.is_relative_to(Path(directory).resolve()), 'Witness snapshot escapes its run directory'
    assert path.is_file(), 'Declared geometric witness snapshot is missing'
    digest = sha256(path.read_bytes()).hexdigest()
    assert digest == manifest['witness_sha256'], 'Frozen geometric witness cache hash mismatch'
    with np.load(path) as cache:
        q = cache['q']
        assert q.ndim == 2 and q.shape[1] == 23 and np.all(np.isfinite(q))
        count = len(q)
        for key, shape in (('vertex', (count,)), ('goal', (count,)),
                           ('pose', (count, 7)), ('tips', (count, 4, 3))):
            assert cache[key].shape == shape and np.all(np.isfinite(cache[key])), key
        for key in ('vertex', 'goal'):
            assert np.issubdtype(cache[key].dtype, np.integer), key
            assert len(np.unique(cache[key])) == count, key
        with np.load(Path(directory)/'surface_graph.npz') as graph:
            goals = cache['goal']
            assert np.all((goals >= 0) & (goals < len(graph['required_vertices'])))
            np.testing.assert_array_equal(cache['vertex'], graph['required_vertices'][goals])
    return dict(geometric_witness_snapshot_verified=True, geometric_witness_snapshot_file=name,
                geometric_witness_snapshot_sha256=digest, geometric_witness_count=count,
                witness_snapshot_scope='Frozen geometric planning candidates; physical coverage still comes only from sensors',
                witness_joint_values_count_as_physical_contact=False)


def _coverage(directory, manifest, summary, trace):
    spec = manifest['coverage']
    expected = dict(radius_m=.025, residual_limit_m=.005, normal_cosine=.5, force_threshold_n=.2)
    assert all(spec.get(key) == value for key, value in expected.items())
    # Rebuild the original complete task surface, not a reachable submesh.
    graph = SurfaceGraph(task_mesh(), coverage_radius_m=.025)
    with np.load(directory/'surface_graph.npz') as saved:
        np.testing.assert_allclose(saved['vertices'], graph.vertices_m, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(saved['faces'], graph.faces)
        np.testing.assert_array_equal(saved['required_vertices'], graph.required_vertices)
    assert len(graph.vertices_m) == 8192 and graph.required_goal_count == 388
    area = np.zeros(len(graph.vertices_m))
    np.add.at(area, graph.faces.ravel(), np.repeat(graph.mesh.area_faces/3., 3))
    mask = np.zeros(len(area), dtype=bool)
    first_touch = np.full(len(area), np.nan)
    rows, fingers = np.nonzero(trace['active'])
    assert np.all(trace['forces_n'][rows, fingers] >= .2)
    points = trace['contact_positions_world_m'][rows, fingers]-np.asarray(manifest['object_position_m'])
    normals = trace['normals_world'][rows, fingers]
    assert np.all(np.isfinite(points)) and np.all(np.isfinite(normals))
    vertices = np.empty(len(rows), dtype=int)
    valid = np.zeros(len(rows), dtype=bool)
    if len(rows):
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(norms, 1., atol=1e-10, rtol=0)
        distances, candidates = graph._tree.query(points, k=24)
        alignments = np.einsum('nkj,nj->nk', graph.normals[candidates], normals/norms[:, None])
        eligible = distances <= distances[:, :1]+.0015
        column = np.argmax(np.where(eligible, alignments, -np.inf), axis=1)
        vertices[:] = candidates[np.arange(len(rows)), column]
        residual = distances[np.arange(len(rows)), column]
        alignment = np.einsum('ij,ij->i', graph.normals[vertices], normals)
        valid = (residual <= .005) & (alignment >= .5)

    @lru_cache(maxsize=8192)
    def neighborhood(vertex):
        distances = dijkstra(graph.adjacency, directed=False, indices=int(vertex), limit=.025)
        return np.flatnonzero(np.isfinite(distances))

    curve = np.zeros(len(trace['time_s']))
    cursor, covered_area = 0, 0.
    for row in range(len(curve)):
        while cursor < len(rows) and rows[cursor] == row:
            if valid[cursor]:
                touched = neighborhood(int(vertices[cursor]))
                added = touched[~mask[touched]]
                first_touch[added] = trace['time_s'][row]
                covered_area += float(area[added].sum())
                mask[touched] = True
            cursor += 1
        curve[row] = covered_area/area.sum()
    fraction = float(area@mask/area.sum())
    np.testing.assert_array_equal(mask, trace['covered_vertices'])
    np.testing.assert_array_equal(mask[graph.required_vertices], trace['required_goal_visited'])
    np.testing.assert_allclose(curve, trace['coverage'], atol=1e-10, rtol=0)
    assert summary['covered_vertex_count'] == int(mask.sum())
    assert summary['visited_required_goals'] == int(mask[graph.required_vertices].sum())
    assert abs(summary['final_covered_area_fraction']-fraction) < 1e-10
    assert summary['vertex_count'] == len(mask)
    assert summary['required_goal_count'] == graph.required_goal_count
    complete = bool(summary['full_surface_complete'])
    complete_stop = summary['stop_reason'] == 'FULL_MEASURED_SURFACE'
    assert complete == complete_stop, 'Full-surface flag and completion stop reason disagree'
    stable_span = 0.
    if complete:
        assert mask.all() and mask[graph.required_vertices].all()
        stability = float(summary.get('completion_stability_s', .15))
        assert np.isfinite(stability) and stability >= .15, 'Completion requires >=150 ms safety hold'
        tail = trace['time_s'] >= trace['time_s'][-1]-stability-1e-9
        stable_span = float(trace['time_s'][tail][-1]-trace['time_s'][tail][0])
        assert stable_span >= stability-1e-9, 'Insufficient recorded final completion-hold duration'
        assert np.all(curve[tail] >= 1.-1e-10), 'Completion hold begins before full measured coverage'
        assert np.all(trace['active'][tail].any(axis=1)), 'Completion hold loses all physical contacts'
        assert np.all(trace['forces_n'][tail].max(axis=1) <= 8.), 'Unsafe force during completion hold'
    np.savez_compressed(directory/'coverage_independent.npz', covered_vertices=mask,
                        first_touch_time_s=first_touch, measured_coverage=curve)
    return dict(actual_contact_samples=int(len(rows)), accepted_contact_samples=int(valid.sum()),
                accepted_contact_anchor_vertices=int(len(np.unique(vertices[valid]))),
                vertices=int(mask.sum()), total_vertices=len(mask),
                required_goals=int(mask[graph.required_vertices].sum()),
                total_required_goals=graph.required_goal_count,
                all_surface_measured=bool(mask.all()), area_fraction=fraction, coverage_radius_m=.025,
                completion_hold_verified=complete, recorded_completion_hold_s=stable_span)


def _active_finger_checks(directory, manifest, trace):
    """Keep geometric finger-search commands separate from physical contact.

    Older passive/grace runs have no search telemetry and remain supported.
    Active geometric seeking is a new B-only information scope, not evidence
    for the old wrist-only versus whole-hand matched information experiment.
    """
    base_source_name = 'Module/i04_geometric_traversal/active_fingers.py'
    source_name = manifest.get('active_finger_source', base_source_name)
    assert isinstance(source_name, str)
    if 'finger_search_active' not in trace:
        assert source_name not in manifest['source_hashes'], 'Active-finger runtime lacks search telemetry'
        return dict(active_finger_search_audited=False,
                    controller_information_scope='Legacy wrist-only or whole-hand target experiment')
    assert source_name in manifest['source_hashes'], 'Active-finger implementation is absent from source snapshot'
    assert base_source_name in manifest['source_hashes'], 'Inherited active-finger configuration is not frozen'
    assert manifest['cell'] == 'B-MCC', 'Geometry-guided finger search is only declared for the new B scope'
    source = ast.parse((directory/'source_snapshot'/base_source_name).read_text(encoding='utf-8'))
    definition = next(node for node in source.body
                      if isinstance(node, ast.ClassDef) and node.name == 'ActiveFingerSearchConfig')
    config = {node.target.id: ast.literal_eval(node.value) for node in definition.body
              if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None}
    overrides = manifest.get('active_finger_search', {})
    if isinstance(overrides, dict):
        config.update({key: value for key, value in overrides.items() if key in config})
    seeking = np.asarray(trace['finger_search_active'])
    assert seeking.dtype == bool and seeking.shape == trace['active'].shape
    assert seeking.shape[1:] == (4,)
    assert not np.any(seeking[trace['forces_n'].max(axis=1) >= 8.]), \
        'Finger search must pause during high-force release'
    assert not np.any(seeking & trace['active']), 'Seeking state cannot masquerade as measured contact'
    if 'finger_search_enabled' in trace:
        enabled = np.asarray(trace['finger_search_enabled'], dtype=bool)
        assert enabled.shape in ((len(seeking),), seeking.shape)
        if enabled.ndim == 1:
            enabled = enabled[:, None]
        assert not np.any(seeking & ~enabled), 'Disabled phase still commands finger seeking'
    if 'finger_search_measured_active' in trace:
        np.testing.assert_array_equal(trace['finger_search_measured_active'], trace['active'])
    if 'finger_search_state' in trace:
        state = np.asarray(trace['finger_search_state'])
        allowed = ('DISABLED', 'CONTACT_HOLD', 'FORCE_PAUSE', 'DEBOUNCE', 'OUT_OF_REACH',
                   'SEARCH_LIMIT', 'GEOMETRIC_TARGET_NO_CONTACT', 'SEEK', 'JOINT_LIMIT')
        if source_name != base_source_name:
            allowed += ('INTENTIONAL_AIR', 'MASKED_HOLD', 'INFEASIBLE_HOLD_REBASE_REQUIRED')
        if manifest.get('protocol') == CONTEXT_PROTOCOLS[3]:
            assert source_name == 'Module/i04_geometric_traversal/active_fingers_v6.py'
            allowed += ('NO_LEGAL_INWARD_STEP',)
        assert state.shape == seeking.shape and np.isin(state, allowed).all()
        np.testing.assert_array_equal(state == 'SEEK', seeking)
    if 'finger_search_loss_age_s' in trace:
        loss_age = np.asarray(trace['finger_search_loss_age_s'])
        assert loss_age.shape == seeking.shape
        assert np.all(np.isfinite(loss_age)) and np.all(loss_age >= -1e-12)
        assert np.all(loss_age[seeking] >= config['loss_debounce_s']-1e-10)
        # V3 preserves physical loss history across rebase and gives each
        # explicit safe posture attempt its own bounded local-search clock.
        clock_name = 'finger_search_elapsed_s' if 'finger_search_elapsed_s' in trace else 'finger_search_loss_age_s'
        assert np.all(trace[clock_name][seeking] < config['maximum_search_time_s']+1e-10)
    if 'finger_search_recontacted' in trace:
        recovered = np.asarray(trace['finger_search_recontacted'], dtype=bool)
        assert recovered.shape == seeking.shape and not np.any(recovered & ~trace['active'])
    maximum_distance = None
    if 'finger_search_distance_m' in trace:
        distance = np.asarray(trace['finger_search_distance_m'])
        assert distance.shape == seeking.shape
        assert np.all(np.isfinite(distance)) and np.all(distance >= -1e-12)
        maximum_distance = float(distance.max())
        assert maximum_distance <= config['maximum_search_distance_m']+1e-10, \
            'Finger search exceeded declared distance bound'
    held_mask = np.zeros((len(seeking), 16), dtype=bool)
    held_rows = 0
    if 'finger_search_state' in trace:
        held = np.asarray(trace['finger_search_state']) == 'INFEASIBLE_HOLD_REBASE_REQUIRED'
        if np.any(held):
            nominal_name = next((key for key in ('finger_search_nominal_rad', 'source_nominal_rad') if key in trace), None)
            assert nominal_name is not None and 'finger_nominal_rad' in trace, \
                'An infeasible held reference requires independent nominal/proposal telemetry'
            assert trace[nominal_name].shape == (len(seeking), 23)
            assert trace['finger_nominal_rad'].shape == (len(seeking), 23)
            for finger, definition in enumerate(FINGERS):
                rows = np.flatnonzero(held[:, finger])
                columns = np.asarray([int(name) for name in definition.joint_names])
                if not len(rows):
                    continue
                held_mask[np.ix_(rows, columns)] = True
                assert not seeking[rows, finger].any()
                selected = np.ix_(rows, columns+7)
                np.testing.assert_allclose(trace['finger_nominal_rad'][selected],
                    trace['sensor_command_rad'][selected], atol=1e-12, rtol=0)
                for bias_name in ('finger_search_joint_bias_rad', 'finger_search_bias_rad'):
                    if bias_name in trace:
                        np.testing.assert_allclose(trace[bias_name][np.ix_(rows, columns)],
                            trace['sensor_command_rad'][selected]-trace[nominal_name][selected], atol=1e-12, rtol=0)
                if 'finger_search_delta_rad' in trace:
                    assert not np.any(trace['finger_search_delta_rad'][np.ix_(rows, columns)])
            held_rows = int(held.sum())
    for name in ('finger_search_joint_bias_rad', 'finger_search_bias_rad', 'finger_search_delta_rad'):
        if name in trace:
            assert trace[name].shape == (len(seeking), 16), name
            assert np.all(np.isfinite(trace[name])), name
            limit = (config['maximum_search_joint_speed_rad_s']*config['dt_s']
                     if name == 'finger_search_delta_rad' else config['maximum_joint_bias_rad'])
            # The explicit infeasible branch holds the already-applied servo
            # reference and requests rebase. Its bias may exceed the search
            # envelope, but only after exact prior-control identity is proven.
            checked = np.ones_like(held_mask) if name == 'finger_search_delta_rad' else ~held_mask
            assert np.all(abs(trace[name][checked]) <= limit+1e-10), name
    valid = (np.asarray(trace['finger_search_target_valid'], dtype=bool)
             if 'finger_search_target_valid' in trace else seeking)
    assert valid.shape == seeking.shape
    for target_name in ('finger_search_target_positions_world_m', 'finger_search_target_world_m'):
        if target_name in trace:
            assert trace[target_name].shape == (len(seeking), 4, 3)
            assert np.all(np.isfinite(trace[target_name][valid]))
    normal_name = 'finger_search_target_normals_world'
    if normal_name in trace:
        assert trace[normal_name].shape == (len(seeking), 4, 3)
        target_norm = np.linalg.norm(trace[normal_name][valid], axis=1)
        np.testing.assert_allclose(target_norm, 1., atol=1e-8, rtol=0)
    lifecycle = _active_finger_lifecycle_checks(directory, manifest, trace)
    return dict(active_finger_search_audited=True, audited_active_finger_source=source_name,
                controller_information_scope='B-only geometry-guided active finger planning',
                active_finger_search_samples=int(seeking.sum()),
                fingers_with_active_search=int(np.count_nonzero(seeking.any(axis=0))),
                maximum_logged_finger_search_distance_m=maximum_distance,
                prior_control_hold_finger_samples_verified=held_rows,
                active_finger_config_verified=config,
                search_targets_count_as_contact=False, **lifecycle)


def _measured_recovery_context_checks(directory, manifest, trace, events):
    """Verify context tokens, complete pose history and native support evidence.

    This authority applies only to the explicitly versioned context protocol.
    It permits a finite quota renewal, never a physical-loss-history reset.
    """
    assert manifest['protocol'] in CONTEXT_PROTOCOLS
    source = 'Module/i04_geometric_traversal/recovery_context.py'
    assert source in manifest['source_hashes']
    assert sha256((directory/'source_snapshot'/source).read_bytes()).hexdigest() == manifest['source_hashes'][source]
    cfg = manifest['coordinated_recovery']['measured_geometry_context']
    assert cfg['dt_s'] == .002 and cfg['minimum_translation_m'] >= .04 and cfg['minimum_rotation_deg'] >= 20.
    assert cfg['context_cooldown_s'] >= 20. and cfg['minimum_unresolved_loss_s'] >= 3.
    assert cfg['minimum_stable_support_s'] >= .15 and cfg['stable_recontact_s'] >= .15
    assert cfg['active_force_min_n'] >= .2 and cfg['safe_force_max_n'] <= 8.
    assert 1 <= cfg['max_requests_per_context'] <= 3
    kinds = ('MEASURED_LOSS_EPISODE_STARTED', 'MEASURED_LOSS_EPISODE_RESOLVED',
             'MEASURED_RECOVERY_REQUEST_RECORDED', 'MEASURED_RECOVERY_CONTEXT_GRANTED')
    journal = [e for e in events if e['kind'] in kinds]
    histories, starts = [[] for _ in range(4)], [None]*4
    episodes, contexts, requests = np.zeros(4, int), np.zeros(4, int), np.zeros(4, int)
    times = np.asarray(trace['time_s'])
    expected_ids = np.zeros(trace['active'].shape, int)
    expected_requests = np.zeros(trace['active'].shape, int)
    grant_times = [[] for _ in range(4)]
    native_frames, native_max_error, grant_count = 0, 0., 0
    context = None
    request_events = {(float(e['time_s']), int(e['finger'])): e for e in events
                      if e['kind'] == 'FINGER_POSTURE_RECOVERY_REQUEST'}
    for sequence, event in enumerate(journal, 1):
        finger, now, kind = int(event['finger']), float(event['time_s']), event['kind']
        assert finger in range(4) and event['event_id'] == sequence
        assert not event['contact_inferred'] and not event['coverage_credit']
        if kind == 'MEASURED_LOSS_EPISODE_STARTED':
            assert starts[finger] is None and not histories[finger]
            episodes[finger] += 1
            starts[finger] = now
            histories[finger].append(dict(time_s=now, kind='INITIAL_MEASURED_LOSS', palm_pose_world=event['palm_pose_world']))
        elif kind == 'MEASURED_LOSS_EPISODE_RESOLVED':
            assert starts[finger] is not None and event['stable_safe_contact_s'] >= cfg['stable_recontact_s']-1e-10
            assert event['consumed_poses'] == histories[finger]
            assert abs(event['unresolved_duration_s']-(now-starts[finger])) < 1e-9
            histories[finger], starts[finger], requests[finger] = [], None, 0
        elif kind == 'MEASURED_RECOVERY_REQUEST_RECORDED':
            assert starts[finger] is not None and event['accepted'] and not event['path_authorized']
            assert (now, finger) in request_events
            request = request_events[(now, finger)]
            requests[finger] += 1
            assert event['requests_this_context'] == requests[finger] <= cfg['max_requests_per_context']
            assert request['request_number_this_context'] == requests[finger]
            assert event['requests_this_episode'] == request['request_number_this_loss_episode']
            histories[finger].append(dict(time_s=now, kind='MONITOR_REQUEST', palm_pose_world=event['palm_pose_world']))
        else:
            assert starts[finger] is not None and event['accepted'] and event['budget_exhausted']
            assert not event['clear_loss_history'] and not event['path_authorized'] and not event['measured_recovery_success']
            assert event['reopen_local_search_budget'] and event['budget_effect'] == 'NEXT_NATIVE_TICK'
            assert event['consumed_poses_before'] == histories[finger] and histories[finger]
            assert event['previous_context_requests'] == requests[finger]
            assert now-starts[finger] >= cfg['minimum_unresolved_loss_s']-1e-9
            assert abs(event['unresolved_loss_age']-(now-starts[finger])) < 1e-9
            assert now-histories[finger][-1]['time_s'] >= cfg['context_cooldown_s']-1e-9
            pose = np.asarray(event['palm_pose_world'], float)
            np.testing.assert_allclose(np.linalg.norm(pose[3:]), 1., atol=1e-10, rtol=0)
            for previous in histories[finger]:
                old = np.asarray(previous['palm_pose_world'], float)
                distance = float(np.linalg.norm(pose[:3]-old[:3]))
                angle = float(np.degrees(2*np.arccos(np.clip(abs(pose[3:]@old[3:]), 0., 1.))))
                assert distance >= cfg['minimum_translation_m']-1e-10 or angle >= cfg['minimum_rotation_deg']-1e-8, 'Previously consumed pose renewed quota'
            budget, search_cfg = event['budget_before'], manifest['active_finger_search']
            cap = manifest['active_finger_lifecycle']['maximum_posture_attempts_per_measured_context']
            assert (budget['search_elapsed_s'] >= search_cfg['maximum_search_time_s']-1e-10
                or budget['search_distance_m'] >= search_cfg['maximum_search_distance_m']-1e-10
                or budget['attempts_this_context'] >= cap or budget['request_budget_remaining'] <= 0), 'Unexhausted quota renewed'
            contexts[finger] += 1
            for name in ('search_consumer', 'participation_consumer'):
                consumer = event[name]
                assert consumer['accepted'] and consumer['finger'] == finger and consumer['context_id'] == contexts[finger]
            assert event['search_consumer']['loss_age_s'] >= event['loss_age_s']-1e-9
            assert event['participation_consumer']['requests_issued_this_loss_episode'] == event['requests_this_episode']
            window_name = event['native_support_window']
            path = (directory/window_name).resolve()
            assert path.is_relative_to(directory.resolve()) and path.is_file()
            assert sha256(path.read_bytes()).hexdigest() == event['native_support_window_sha256']
            with np.load(path) as saved: window = {key:saved[key].copy() for key in saved.files}
            wt = window['time_s']
            np.testing.assert_allclose(np.diff(wt), .002, atol=1e-10, rtol=0)
            assert len(wt) >= 76 and abs(wt[-1]-now) < 1e-10 and wt[-1]-wt[0] >= cfg['minimum_stable_support_s']-1e-10
            # Link the dense evidence back to every overlapping main-trace
            # sample, rather than accepting a physically valid unrelated clip.
            positions = np.searchsorted(times, wt-1e-9)
            overlaps = [row for row, index in enumerate(positions)
                        if index < len(times) and abs(times[index]-wt[row]) < 1e-9]
            if times[0] <= wt[0]+1e-9 and times[-1] >= wt[-1]-1e-9:
                assert len(overlaps) >= 15
            for row in overlaps:
                index = positions[row]
                for key in ('q_rad','qvel_rad_s','sensor_command_rad','qacc_warmstart',
                            'palm_pose_world','active','forces_n'):
                    np.testing.assert_array_equal(window[key][row],trace[key][index])
            support = int(event['support_finger'])
            assert support in range(4) and support != finger
            assert np.all(window['active'][:,support])
            assert np.all(window['forces_n'][:,support] >= cfg['active_force_min_n'])
            assert np.all(window['forces_n'] < cfg['safe_force_max_n'])
            assert all(phase in cfg['surface_phases'] for phase in window['phase'])
            assert not window['active'][-1,finger] and not event['active'][finger]
            np.testing.assert_allclose(window['palm_pose_world'][-1], pose, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(window['active'][-1], event['active'])
            np.testing.assert_array_equal(window['forces_n'][-1], event['forces_n'])
            if context is None: context = _native_context(directory)
            model, data = context.model, context.data
            _restore_native(context, {key:window[key][0] for key in ('q_rad','qvel_rad_s','sensor_command_rad','qacc_warmstart')})
            data.time = float(window['native_time_s'][0])
            palm_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'fr3_palm_control_site')
            assert palm_site >= 0
            for row in range(len(wt)):
                force, _, _, active, unsafe, _ = contact_measurement(context.handles, data, .2)
                pairs = ((data.qpos[context.qpos],window['q_rad'][row]),
                    (data.qvel[context.dof],window['qvel_rad_s'][row]),
                    (data.ctrl[context.actuator_ids],window['sensor_command_rad'][row]),
                    (data.qacc_warmstart,window['qacc_warmstart'][row]),
                    (data.time,window['native_time_s'][row]), (force,window['forces_n'][row]),
                    (data.site_xpos[palm_site],window['palm_pose_world'][row,:3]))
                native_max_error = max(native_max_error, *(float(np.max(abs(np.asarray(a)-b))) for a,b in pairs))
                np.testing.assert_array_equal(active,window['active'][row])
                assert not unsafe and max([0.]+[c['force_n'] for c in _native_arm_contacts(model,data)]) <= .5
                rotation = np.empty(4); mujoco.mju_mat2Quat(rotation,data.site_xmat[palm_site])
                assert abs(abs(rotation@window['palm_pose_world'][row,3:])-1.) < 1e-9
                # The next observation's prior control is exactly the control
                # that advanced this current state; no intermediate reset.
                if row+1 < len(wt):
                    data.ctrl[context.actuator_ids] = window['sensor_command_rad'][row+1]
                    mujoco.mj_step(model,data); mujoco.mj_forward(model,data)
            assert native_max_error < 1e-7, native_max_error
            native_frames += len(wt); grant_count += 1
            histories[finger].append(dict(time_s=now,kind='GRANTED_RECOVERY_CONTEXT',palm_pose_world=event['palm_pose_world']))
            requests[finger] = 0
            expected_ids[times >= now-1e-10,finger] += 1
            grant_times[finger].append(now)
        assert event['episode_id'] == episodes[finger] and event['context_id'] == contexts[finger]
        expected_requests[times >= now-1e-10,finger] = requests[finger]
    np.testing.assert_array_equal(expected_ids,trace['finger_recovery_context_id'])
    np.testing.assert_array_equal(expected_requests,trace['finger_recovery_context_requests'])
    assert len([e for e in journal if e['kind']=='MEASURED_RECOVERY_REQUEST_RECORDED']) == len(request_events)
    for field, before in (('finger_search_next_elapsed_s','finger_search_elapsed_s'),('finger_search_next_distance_m','finger_search_distance_m')):
        expected = np.asarray(trace[before]).copy()
        for finger, values in enumerate(grant_times):
            for now in values: expected[np.isclose(times,now,atol=1e-10,rtol=0),finger] = 0.
        np.testing.assert_allclose(trace[field],expected,atol=1e-10,rtol=0)
    return dict(measured_context_grants=grant_count,context_journal_events=len(journal),
                context_native_support_frames=native_frames,context_native_replay_max_error=native_max_error,
                context_scope='Explicit exhausted-quota renewal; physical loss episode survives'), grant_times


def _endpoint_escalation_checks(directory, manifest, trace, events):
    """Independently validate finite endpoint permission and native support.

    This does not execute the policy being audited and does not turn a failed
    finite IK search into a reachability claim or a recovery success.
    """
    assert manifest['protocol'] in CONTEXT_PROTOCOLS[2:]
    source='Module/i04_geometric_traversal/endpoint_recovery_policy.py'
    assert source in manifest['source_hashes']
    assert sha256((directory/'source_snapshot'/source).read_bytes()).hexdigest()==manifest['source_hashes'][source]
    cfg=manifest['coordinated_recovery']['anchored_endpoint_escalation']
    assert cfg['dt_s']==.002 and cfg['minimum_unresolved_loss_s']>=30.
    assert cfg['minimum_failed_contexts']>=2 and cfg['escalation_cooldown_s']>=60.
    assert cfg['stable_support_s']>=.15 and cfg['stable_recontact_s']>=.15
    assert cfg['active_force_min_n']>=.2 and cfg['safe_force_max_n']<=8.
    assert set(cfg['support_phases'])<=set(('SLIDE','SELECT','LOCAL_SEARCH','REACQUIRE'))
    assert cfg['qualifying_failure_reasons']==['NO_SAFE_PATH_AMONG_BOUNDED_SURFACE_TARGETS']
    dispatches={e['request_id']:e for e in events if e['kind']=='RECOVERY_REQUEST_TAKEN_AT_BOUNDARY'}
    plans={e['request']['request_id']:e for e in events if e['kind']=='COORDINATED_RECOVERY_PLAN'}
    failures=[[] for _ in range(4)];starts=[None]*4;episodes=np.zeros(4,int);contexts=np.zeros(4,int)
    used=set();grants={};escalated=[set() for _ in range(4)];last=np.full(4,-np.inf)
    sequence=0;native_frames=0;native_error=0.;native=None;times=np.asarray(trace['time_s'])
    for event in events:
        kind=event['kind']
        if kind in ('MEASURED_LOSS_EPISODE_STARTED','MEASURED_LOSS_EPISODE_RESOLVED','MEASURED_RECOVERY_CONTEXT_GRANTED'):
            finger=int(event['finger']);contexts[finger]=event['context_id'];episodes[finger]=event['episode_id']
            if kind=='MEASURED_LOSS_EPISODE_STARTED':starts[finger]=event['time_s']
            elif kind=='MEASURED_LOSS_EPISODE_RESOLVED':
                failures[finger]=[];starts[finger]=None;escalated[finger].clear()
            continue
        if not kind.startswith('ENDPOINT_RECOVERY_ESCALATION_'):continue
        assert kind in ('ENDPOINT_RECOVERY_ESCALATION_GRANTED','ENDPOINT_RECOVERY_ESCALATION_DENIED'), 'Unexpected stale policy call'
        sequence+=1;assert event['event_id']==sequence
        finger=int(event['finger']);now=float(event['time_s']);rid=event['request_id']
        assert rid in dispatches and rid not in used;used.add(rid)
        dispatch=dispatches[rid];assert dispatch['mode']=='SHORT_FIXED_WRIST'
        assert dispatch['natural_stroke_endpoint'] and dispatch['phase']=='SELECT'
        assert abs(now-dispatch['time_s']-.002)<1e-10 and event['dispatch_time_s']==dispatch['time_s']
        assert dispatch['finger']==finger and event['episode_id']==episodes[finger] and event['context_id']==contexts[finger]
        assert event['episode_id']==dispatch['episode_id'] and event['context_id']==dispatch['context_id']
        for key in ('search_budget_renewed','request_budget_renewed','clear_loss_history','contact_inferred','coverage_credit'):
            assert event[key] is False
        plan=plans[rid];fixed=plan['finger_plan'];current=event['current_fixed_failure']
        assert plan['time_s']==now and not fixed.get('accepted',False)
        assert fixed['reason'] in cfg['qualifying_failure_reasons']
        assert current['reason']==fixed['reason'] and current['time_s']==now and current['request_id']==rid
        assert current['finger']==finger and current['episode_id']==episodes[finger] and current['context_id']==contexts[finger]
        failures[finger].append(current)
        assert event['failed_attempts']==failures[finger], 'Lost, invented, or prematurely reset failure history'
        distinct=sorted({f['context_id'] for f in failures[finger]})
        assert event['distinct_failed_context_ids']==distinct and starts[finger] is not None
        age=now-starts[finger];assert abs(event['unresolved_loss_age_s']-age)<1e-8
        expected_previous=None if not np.isfinite(last[finger]) else float(last[finger])
        assert event['previous_escalation_time_s']==expected_previous
        cooldown=max(0.,cfg['escalation_cooldown_s']-(now-last[finger]))
        assert abs(event['escalation_cooldown_remaining_s']-cooldown)<1e-8
        assert event['accepted']==(kind=='ENDPOINT_RECOVERY_ESCALATION_GRANTED')
        if not event['accepted']:
            assert plan.get('endpoint_escalation_grant_id') is None
            continue
        assert age>=cfg['minimum_unresolved_loss_s']-1e-9 and len(distinct)>=cfg['minimum_failed_contexts']
        assert cooldown<=1e-9 and contexts[finger] not in escalated[finger]
        assert plan['endpoint_escalation_grant_id']==sequence
        path=(directory/event['native_support_window_file']).resolve()
        assert path.is_relative_to(directory.resolve()) and path.is_file()
        assert sha256(path.read_bytes()).hexdigest()==event['native_support_window_sha256']
        with np.load(path) as saved:window={key:saved[key].copy() for key in saved.files}
        wt=window['time_s'];assert len(wt)==76 and abs(wt[-1]-now)<1e-10
        np.testing.assert_allclose(np.diff(wt),.002,atol=1e-10,rtol=0)
        assert wt[-1]-wt[0]>=cfg['stable_support_s']-1e-10
        assert not window['active'][-1,finger] and np.all(window['forces_n']<cfg['safe_force_max_n'])
        assert all(p in cfg['support_phases'] for p in window['phase']) and window['phase'][-1]=='SELECT'
        support=[f for f in range(4) if f!=finger and np.all(window['active'][:,f])
                 and np.all(window['forces_n'][:,f]>=cfg['active_force_min_n'])]
        assert support and event['stable_support_fingers']==support
        rows=event['native_support_window'];assert len(rows)==len(wt)
        for i,row in enumerate(rows):
            assert row['time_s']==wt[i] and row['phase']==window['phase'][i]
            for key in ('active','forces_n','palm_pose_world'):
                np.testing.assert_allclose(row[key],window[key][i],atol=1e-12,rtol=0)
        np.testing.assert_allclose(current['palm_pose_world'],window['palm_pose_world'][-1],atol=1e-12,rtol=0)
        # Link dense native evidence to every overlapping 100 Hz main sample.
        overlaps=0
        for i,t in enumerate(wt):
            index=int(np.searchsorted(times,t-1e-9))
            if index<len(times) and abs(times[index]-t)<1e-9:
                overlaps+=1
                for key in ('q_rad','qvel_rad_s','sensor_command_rad','qacc_warmstart','palm_pose_world','active','forces_n'):
                    np.testing.assert_array_equal(window[key][i],trace[key][index])
        if times[0]<=wt[0]+1e-9 and times[-1]>=wt[-1]-1e-9:assert overlaps>=15
        if native is None:native=_native_context(directory)
        model,data=native.model,native.data
        _restore_native(native,{key:window[key][0] for key in ('q_rad','qvel_rad_s','sensor_command_rad','qacc_warmstart')})
        data.time=float(window['native_time_s'][0])
        palm=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'fr3_palm_control_site');assert palm>=0
        for i in range(len(wt)):
            force,_,_,active,unsafe,_=contact_measurement(native.handles,data,.2)
            pairs=((data.qpos[native.qpos],window['q_rad'][i]),(data.qvel[native.dof],window['qvel_rad_s'][i]),
                (data.ctrl[native.actuator_ids],window['sensor_command_rad'][i]),(data.qacc_warmstart,window['qacc_warmstart'][i]),
                (data.time,window['native_time_s'][i]),(force,window['forces_n'][i]),(data.site_xpos[palm],window['palm_pose_world'][i,:3]))
            native_error=max(native_error,*(float(np.max(abs(np.asarray(a)-b))) for a,b in pairs))
            np.testing.assert_array_equal(active,window['active'][i]);assert not unsafe
            assert max([0.]+[c['force_n'] for c in _native_arm_contacts(model,data)])<=.5
            rotation=np.empty(4);mujoco.mju_mat2Quat(rotation,data.site_xmat[palm])
            assert abs(abs(rotation@window['palm_pose_world'][i,3:])-1.)<1e-9
            if i+1<len(wt):
                data.ctrl[native.actuator_ids]=window['sensor_command_rad'][i+1]
                mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
        native_frames+=len(wt);last[finger]=now;escalated[finger].add(contexts[finger]);grants[sequence]=event
    assert native_error<1e-7,native_error
    return dict(endpoint_escalation_audited=True,endpoint_policy_considerations=sequence,
                endpoint_escalation_grants=len(grants),endpoint_native_support_frames=native_frames,
                endpoint_native_replay_max_error=native_error),grants


def _boundary_queue_checks(directory, manifest, trace, events):
    """Audit one-shot deferred requests and physical roots of endpoint paths."""
    assert manifest['protocol'] in CONTEXT_PROTOCOLS[1:]
    escalation_result, escalation_grants = ({}, {})
    if manifest['protocol'] in CONTEXT_PROTOCOLS[2:]:
        escalation_result,escalation_grants=_endpoint_escalation_checks(directory,manifest,trace,events)
    else:
        assert not any(e['kind'].startswith('ENDPOINT_RECOVERY_ESCALATION_') for e in events)
    source = 'Module/i04_geometric_traversal/recovery_queue.py'
    assert source in manifest['source_hashes']
    assert sha256((directory/'source_snapshot'/source).read_bytes()).hexdigest() == manifest['source_hashes'][source]
    cfg = manifest['coordinated_recovery']['deferred_request_queue']
    assert cfg['dt_s'] == .002 and cfg['maximum_endpoint_path_duration_s'] <= 8.
    assert cfg['stable_recontact_s'] >= .15 and cfg['safe_force_max_n'] <= 8.
    requests = {e['request_id']:e for e in events if e['kind']=='FINGER_POSTURE_RECOVERY_REQUEST'}
    queue_kinds = ('RECOVERY_REQUEST_QUEUED','RECOVERY_REQUEST_DROPPED',
                   'RECOVERY_REQUEST_TAKEN_AT_BOUNDARY','RECOVERY_REQUEST_ENQUEUE_REJECTED')
    journal = [e for e in events if e['kind'] in queue_kinds]
    pending, seen, taken = {}, set(), {}
    expected_pending = np.zeros(trace['active'].shape, bool)
    times = np.asarray(trace['time_s'])
    endpoint_times = [float(e['time_s']) for e in events if e['kind']=='STROKE_ENDPOINT']
    context_events = [e for e in events if e['kind'] in ('MEASURED_LOSS_EPISODE_STARTED',
        'MEASURED_LOSS_EPISODE_RESOLVED','MEASURED_RECOVERY_REQUEST_RECORDED','MEASURED_RECOVERY_CONTEXT_GRANTED')]
    context = None
    native_error, short_dispatches, short_paths = 0., 0, 0

    def current_ids(finger, now):
        previous = [e for e in context_events if e['finger']==finger and e['time_s'] <= now+1e-10]
        return (previous[-1]['episode_id'],previous[-1]['context_id']) if previous else (0,0)

    for sequence, event in enumerate(journal, 1):
        assert event['event_id']==sequence and not event['budget_refunded']
        assert not event['contact_inferred'] and not event['coverage_credit']
        assert event['kind']!='RECOVERY_REQUEST_ENQUEUE_REJECTED', 'Runtime rejected queue input'
        finger, request_id, now = int(event['finger']), int(event['request_id']), float(event['time_s'])
        assert finger in range(4) and request_id in requests
        request = requests[request_id]
        assert request['finger']==finger and event['request_time_s']==request['time_s']
        if event['kind']=='RECOVERY_REQUEST_QUEUED':
            assert event['accepted'] and request_id not in seen and finger not in pending
            assert now==float(request['time_s'])
            assert (event['episode_id'],event['context_id'])==current_ids(finger,now)
            assert request['recovery_context_id']==event['context_id']
            pending[finger]=event;seen.add(request_id)
        else:
            assert finger in pending and pending[finger]['request_id']==request_id
            item=pending.pop(finger)
            assert abs(event['request_age_s']-(now-event['request_time_s'])) < 1e-9
            if event['kind']=='RECOVERY_REQUEST_DROPPED':
                reason=event['reason']
                if reason=='ACTUAL_STABLE_SAME_FINGER_RECONTACT':
                    assert event['stable_safe_contact_s'] >= cfg['stable_recontact_s']-1e-10
                    assert any(e['kind']=='MEASURED_LOSS_EPISODE_RESOLVED' and e['finger']==finger
                               and abs(e['time_s']-now)<1e-9 for e in context_events)
                elif reason=='PHYSICAL_LOSS_EPISODE_CHANGED':
                    assert event['current_episode_id'] != item['episode_id']
                elif reason=='MEASURED_GEOMETRY_CONTEXT_CHANGED':
                    assert event['current_context_id'] != item['context_id']
                elif reason=='REQUEST_TTL_EXPIRED':
                    assert event['request_age_s'] >= cfg['request_ttl_s']-1e-10
                else:raise AssertionError('Unknown queue discard condition')
            else:
                assert event['accepted'] and not event['path_authorized'] and request_id not in taken
                assert event['request']==request
                assert event['request_age_s'] < cfg['request_ttl_s']-1e-12
                assert (event['episode_id'],event['context_id'])==current_ids(finger,now)
                assert not any(abs(other['time_s']-now)<1e-10 for other in taken.values())
                if event['mode']=='SHORT_FIXED_WRIST':
                    assert event['phase']=='SELECT' and event['natural_stroke_endpoint']
                    assert any(abs(now-end)<1e-10 for end in endpoint_times)
                    assert event['fixed_wrist_required'] and event['selected_finger_only']
                    assert event['maximum_path_duration_s']<=8.
                    short_dispatches+=1
                else:
                    assert event['mode']=='FULL_FALLBACK' and event['phase']=='LOCAL_SEARCH'
                state=event['actual_native_state']
                assert not state['active'][finger] and max(state['forces_n']) < 8.
                assert np.any(np.asarray(state['active']) & (np.asarray(state['forces_n']) >= .2))
                assert state['episode_ids'][finger]==event['episode_id'] and state['context_ids'][finger]==event['context_id']
                if context is None:context=_native_context(directory)
                _restore_native(context,state);context.data.time=float(state['native_time_s'])
                force,_,_,active,unsafe,_=contact_measurement(context.handles,context.data,.2)
                native_error=max(native_error,float(np.max(abs(force-state['forces_n']))))
                np.testing.assert_array_equal(active,state['active']);assert not unsafe
                assert max([0.]+[c['force_n'] for c in _native_arm_contacts(context.model,context.data)])<=.5
                taken[request_id]=event
        expected_pending[times >= now-1e-10,finger] = finger in pending
    np.testing.assert_array_equal(expected_pending,trace['finger_recovery_pending'])
    begins=[e for e in events if e['kind']=='COORDINATED_RECOVERY_BEGIN']
    used=set()
    for event in begins:
        request_id=int(event['request_id']);assert request_id in taken and request_id not in used
        used.add(request_id);dispatch=taken[request_id];finger=int(event['selected_finger'])
        assert finger==dispatch['finger'] and abs(event['time_s']-dispatch['time_s']-.002)<1e-10
        assert event['time_s']-requests[request_id]['time_s'] < cfg['request_ttl_s']-1e-12
        escalated=event['dispatch_mode']=='ANCHORED_ENDPOINT_FALLBACK'
        if escalated:
            assert manifest['protocol'] in CONTEXT_PROTOCOLS[2:] and dispatch['mode']=='SHORT_FIXED_WRIST'
            grant=escalation_grants[event['endpoint_escalation_grant_id']]
            assert grant['request_id']==request_id and grant['time_s']==event['time_s'] and grant['finger']==finger
            assert event['path_kind']=='WRIST_PIVOT' and event['anchor_finger'] in grant['stable_support_fingers']
            assert event['duration_s']+2.3<=60. and event['endpoint_motion_pause']
        else:
            assert event['dispatch_mode']==dispatch['mode'] and event.get('endpoint_escalation_grant_id') is None
        path=(directory/event['path_snapshot']).resolve()
        assert path.is_relative_to(directory.resolve()) and path.is_file()
        assert sha256(path.read_bytes()).hexdigest()==event['path_snapshot_sha256']
        with np.load(path) as saved:route={key:saved[key].copy() for key in saved.files}
        assert int(route['selected_finger'])==finger and abs(float(route['durations_s'].sum())-event['duration_s'])<1e-10
        np.testing.assert_array_equal(route['q'][0],route['actual_root_q'])
        np.testing.assert_array_equal(route['actual_root_q'],route['qpos'][context.qpos])
        # Reconstruct dispatch -> next native tick with the next state's real
        # prior control, verifying the plan starts at the physical endpoint.
        _restore_native(context,dispatch['actual_native_state'])
        context.data.time=float(dispatch['actual_native_state']['native_time_s'])
        context.data.ctrl[:]=route['ctrl'];mujoco.mj_step(context.model,context.data);mujoco.mj_forward(context.model,context.data)
        for key in ('qpos','qvel','ctrl','qacc_warmstart'):
            native_error=max(native_error,float(np.max(abs(getattr(context.data,key)-route[key]))))
        assert context.data.time==float(route['native_time_s'])
        force,_,_,active,unsafe,_=contact_measurement(context.handles,context.data,.2)
        assert not active[finger] and force.max()<8. and active.any() and not unsafe
        if dispatch['mode']=='SHORT_FIXED_WRIST' and not escalated:
            assert event['path_kind']=='FINGER_RECONFIGURE' and event['endpoint_motion_pause']
            assert event['duration_s'] <= cfg['maximum_endpoint_path_duration_s']
            selected=np.arange(7+4*finger,11+4*finger);others=np.setdiff1d(np.arange(23),selected)
            np.testing.assert_array_equal(route['q'][:,others],np.tile(route['actual_root_q'][others],(len(route['q']),1)))
            short_paths+=1
    assert native_error<1e-7,native_error
    return dict(boundary_queue_audited=True,queued_recovery_requests=len(seen),
                boundary_dispatches=len(taken),short_fixed_wrist_dispatches=short_dispatches,
                short_fixed_wrist_paths=short_paths,boundary_native_replay_max_error=native_error,
                **escalation_result,
                boundary_audit_scope='Physical natural endpoint, one-shot request; <=8s selected-only path or explicitly verified finite anchored escalation; recovery requires measured contact')


def _active_finger_lifecycle_checks(directory, manifest, trace):
    """Check sampled V3 attempt budgets without equating planning with touch."""
    fields = ('finger_search_elapsed_s', 'finger_contact_stable_age_s',
              'finger_posture_attempt_id', 'finger_posture_attempts_this_episode')
    if not any(key in trace for key in fields):
        return dict(active_finger_lifecycle_audited=False,
                    lifecycle_scope='Legacy development trace has no per-attempt lifecycle telemetry')
    assert all(key in trace for key in fields), 'Incomplete per-attempt lifecycle telemetry'
    spec = manifest['active_finger_lifecycle']
    assert spec['loss_history_survives_rebase'] and spec['air_pauses_local_budget']
    stable_s = float(spec['stable_measured_contact_reset_s'])
    has_contexts = manifest.get('protocol') in CONTEXT_PROTOCOLS
    cap = int(spec['maximum_posture_attempts_per_measured_context' if has_contexts else 'maximum_posture_attempts_per_loss_episode'])
    assert stable_s >= .15 and 1 <= cap <= 3
    shape = trace['active'].shape
    for key in fields:
        value = np.asarray(trace[key])
        assert value.shape == shape and np.all(np.isfinite(value)) and np.all(value >= -1e-12), key
    ids = np.asarray(trace['finger_posture_attempt_id'])
    counts = np.asarray(trace['finger_posture_attempts_this_episode'])
    assert np.issubdtype(ids.dtype, np.integer) and np.issubdtype(counts.dtype, np.integer)
    assert np.all(np.diff(ids, axis=0) >= 0), 'Lifetime posture attempt IDs were reset'
    assert np.all(counts <= ids)
    if not has_contexts:
        assert np.all(counts <= cap)
    events = json.loads((directory/'events.json').read_text(encoding='utf-8'))
    context_result, context_times = ({}, [[] for _ in range(4)])
    if has_contexts:
        assert spec['context_renews_only_exhausted_budget'] and spec['true_loss_history_preserved']
        context_fields = ('finger_recovery_context_id','finger_posture_attempts_this_context',
                          'finger_recovery_context_requests','finger_search_next_elapsed_s','finger_search_next_distance_m')
        for key in context_fields:
            assert key in trace and np.asarray(trace[key]).shape == shape
            assert np.all(np.isfinite(trace[key])) and np.all(trace[key] >= 0)
            if key not in ('finger_search_next_elapsed_s','finger_search_next_distance_m'):
                assert np.issubdtype(np.asarray(trace[key]).dtype,np.integer)
        assert np.all(trace['finger_posture_attempts_this_context'] <= cap)
        assert np.all(trace['finger_recovery_context_requests'] <= 3)
        context_result, context_times = _measured_recovery_context_checks(directory,manifest,trace,events)
    else:
        assert not any(e['kind']=='MEASURED_RECOVERY_CONTEXT_GRANTED' for e in events), 'Unversioned quota renewal'
    if manifest.get('protocol') in CONTEXT_PROTOCOLS[1:]:
        context_result.update(_boundary_queue_checks(directory,manifest,trace,events))
    grants = [e for e in events if e['kind'] == 'COORDINATED_RECOVERY_BEGIN']
    times = np.asarray(trace['time_s'])
    observed_ids = np.zeros(shape, dtype=int)
    grant_times = [[] for _ in range(4)]
    totals = np.zeros(4, dtype=int)
    for event in grants:
        finger = int(event['selected_finger'])
        assert finger in range(4)
        grant = event['local_search_grant']
        assert grant['accepted'] and grant['reason'] == 'EXPLICIT_SAFE_POSTURE_ATTEMPT'
        totals[finger] += 1
        assert grant['attempt_id'] == totals[finger]
        observed_ids[times >= float(event['time_s'])-1e-10, finger] += 1
        grant_times[finger].append(float(event['time_s']))
    np.testing.assert_array_equal(ids, observed_ids)
    ages = np.asarray(trace['finger_contact_stable_age_s'])
    actual_safe = trace['active'] & (trace['forces_n'] < 8.)
    assert np.all(ages[~actual_safe] == 0.)
    assert np.all(ages <= times[:, None]+.002+1e-9)
    reset_checks = 0
    reset_fields = ['finger_search_elapsed_s', 'finger_search_distance_m',
                    'finger_search_loss_age_s', 'finger_posture_attempts_this_episode']
    if has_contexts: reset_fields.extend(('finger_posture_attempts_this_context','finger_recovery_context_requests'))
    for field in reset_fields:
        values = np.asarray(trace[field])
        rows, fingers = np.nonzero(np.diff(values, axis=0) < -1e-10)
        for before, finger in zip(rows, fingers):
            after = before+1
            sample_span = times[after]-times[before]
            stable_reset = max(ages[before, finger], ages[after, finger]) >= stable_s-sample_span-1e-9
            explicit_reset = (field in ('finger_search_elapsed_s', 'finger_search_distance_m')
                and any(times[before]-1e-10 < t <= times[after]+1e-10 for t in grant_times[finger]))
            if has_contexts and field in ('finger_search_elapsed_s','finger_search_distance_m',
                                         'finger_posture_attempts_this_context','finger_recovery_context_requests'):
                explicit_reset |= any(times[before]-1e-10 <= t <= times[after]+1e-10 for t in context_times[finger])
            assert stable_reset or explicit_reset, (field, int(after), int(finger), 'Undeclared episode/budget reset')
            reset_checks += 1
    return dict(active_finger_lifecycle_audited=True,
                lifecycle_scope='Sampled clocks and explicit attempt grants; sensor contact remains independent',
                explicit_safe_posture_attempts=len(grants),
                **({'maximum_attempts_per_measured_context':cap} if has_contexts else {'maximum_attempts_per_loss_episode':cap}),
                **context_result,
                stable_contact_reset_threshold_s=stable_s, lifecycle_reset_transitions_checked=reset_checks,
                maximum_physical_loss_age_s=float(np.max(trace['finger_search_loss_age_s'])),
                maximum_local_search_elapsed_s=float(np.max(trace['finger_search_elapsed_s'])))


def _arm_safety_required(manifest):
    return ('arm_self_collision' in manifest
            or 'Module/i04_geometric_traversal/sliding_safety.py' in manifest['source_hashes']
            or manifest.get('protocol', '').endswith('.v2.1'))


def _native_context(directory):
    model = compile_model_xml((directory/'model.xml').read_text(encoding='utf-8'))
    assert (model.nq, model.nv, model.nu) == (23, 23, 23)

    def ids(kind, names):
        result = np.array([mujoco.mj_name2id(model, kind, name) for name in names])
        assert np.all(result >= 0), names
        return result

    joint_ids = ids(mujoco.mjtObj.mjOBJ_JOINT, (*ARM_JOINT_NAMES, *HAND_JOINT_NAMES))
    actuator_ids = ids(mujoco.mjtObj.mjOBJ_ACTUATOR,
                       (*[f'fr3_arm_act_{i}' for i in range(1, 8)], *[f'act_{i}' for i in range(16)]))
    sites = ids(mujoco.mjtObj.mjOBJ_SITE, [f.pad_site.replace('e05_', 'fr3_') for f in FINGERS])
    geoms = ids(mujoco.mjtObj.mjOBJ_GEOM, [f.proxy_geom.replace('e05_', 'fr3_') for f in FINGERS])
    body = int(ids(mujoco.mjtObj.mjOBJ_BODY, ['fr3_e05_object'])[0])
    object_geoms = np.flatnonzero(model.geom_bodyid == body)
    assert len(object_geoms) >= 256, 'Original object collision pieces are missing'
    handles = SimpleNamespace(model=model, tip_geom_ids=geoms,
                              object_geom_ids=object_geoms, object_geom_id=int(object_geoms[0]))
    return SimpleNamespace(model=model, data=mujoco.MjData(model), handles=handles,
                           qpos=model.jnt_qposadr[joint_ids], dof=model.jnt_dofadr[joint_ids],
                           actuator_ids=actuator_ids, sites=sites)


def _restore_native(context, state):
    model, data = context.model, context.data
    mujoco.mj_resetData(model, data)
    for key in ('q_rad', 'qvel_rad_s', 'sensor_command_rad'):
        value = np.asarray(state[key])
        assert value.shape == (23,) and np.all(np.isfinite(value)), key
    data.qpos[context.qpos] = state['q_rad']
    data.qvel[context.dof] = state['qvel_rad_s']
    data.ctrl[context.actuator_ids] = state['sensor_command_rad']
    if 'qacc_warmstart' in state:
        warmstart = np.asarray(state['qacc_warmstart'])
        assert warmstart.shape == (model.nv,) and np.all(np.isfinite(warmstart))
        data.qacc_warmstart[:] = warmstart
    mujoco.mj_forward(model, data)


def _native_arm_contacts(model, data):
    """Independent native query; do not import the runtime safety predicate."""
    contacts, wrench = [], np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        ga, gb = int(contact.geom1), int(contact.geom2)
        if ga < 0 or gb < 0 or contact.dist > 0.:
            continue
        a, b = int(model.geom_bodyid[ga]), int(model.geom_bodyid[gb])
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ''
                 for body in (a, b)]
        if not all(name.startswith('fr3v2_link') and name[len('fr3v2_link'):].isdigit()
                   for name in names):
            continue
        wa, wb = int(model.body_weldid[a]), int(model.body_weldid[b])
        parent = model.body_parentid
        if (a == b or wa == wb or parent[a] == b or parent[b] == a
                or model.body_weldid[parent[wa]] == wb or model.body_weldid[parent[wb]] == wa):
            continue
        mujoco.mj_contactForce(model, data, index, wrench)
        force = max(0., float(wrench[0]))
        if force > 0.:
            contacts.append(dict(contact_index=index, force_n=force, distance_m=float(contact.dist),
                body_names=names, geom_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                                               for g in (ga, gb)]))
    return contacts


def _physical_checks(directory, trace, manifest=None):
    context = _native_context(directory)
    model, data, handles = context.model, context.data, context.handles
    qpos, dof, actuator_ids = context.qpos, context.dof, context.actuator_ids
    check_arm = manifest is not None and _arm_safety_required(manifest)
    if check_arm:
        assert 'arm_self_force_n' in trace and 'qacc_warmstart' in trace
        arm_logged = np.asarray(trace['arm_self_force_n'])
        assert arm_logged.shape == trace['time_s'].shape
        assert np.all(np.isfinite(arm_logged)) and np.all(arm_logged >= 0.)
    fk_rows = np.unique(np.linspace(0, len(trace['time_s'])-1, 201).astype(int))
    sensor_rows = np.unique(np.linspace(0, len(trace['time_s'])-1, 51).astype(int))
    if check_arm:
        sensor_rows = np.union1d(sensor_rows, [int(np.argmax(arm_logged)),
            int(np.argmax(np.max(trace['forces_n'], axis=1)))])
    max_fk, errors, active_errors, unsafe_rows = 0., np.zeros(3), 0, []
    arm_error, arm_peak, arm_rows = 0., 0., []
    for row in np.union1d(fk_rows, sensor_rows):
        # Sensors were recorded before installing the next controller command.
        keys = ('q_rad', 'qvel_rad_s', 'sensor_command_rad', 'qacc_warmstart')
        _restore_native(context, {key: trace[key][row] for key in keys if key in trace})
        if row in fk_rows:
            max_fk = max(max_fk, float(np.max(abs(data.site_xpos[context.sites]-trace['tip_positions_world_m'][row]))))
        if row in sensor_rows:
            force, point, normal, active, unsafe, _ = contact_measurement(handles, data, .2)
            errors = np.maximum(errors, [np.max(abs(force-trace['forces_n'][row])),
                                         np.max(abs(point-trace['contact_positions_world_m'][row])),
                                         np.max(abs(normal-trace['normals_world'][row]))])
            active_errors += int(np.count_nonzero(active != trace['active'][row]))
            if unsafe:
                unsafe_rows.append(dict(row=int(row), time_s=float(trace['time_s'][row]), reasons=unsafe))
            if check_arm:
                contacts = _native_arm_contacts(model, data)
                value = max([0.] + [contact['force_n'] for contact in contacts])
                arm_error = max(arm_error, abs(value-float(arm_logged[row])))
                arm_peak = max(arm_peak, value)
                if value > .5:
                    arm_rows.append(dict(row=int(row), time_s=float(trace['time_s'][row]), contacts=contacts))
    assert max_fk < 1e-9, max_fk
    # Early pilots did not save solver warmstart and use the zero reset. These
    # tolerances are much smaller than the 0.2 N sensor threshold and 5 mm
    # geometric acceptance residual.
    assert errors[0] <= 1e-5 and errors[1] <= 1e-7 and errors[2] <= 1e-7, errors
    assert active_errors == 0, active_errors
    assert arm_error <= 1e-5, arm_error
    with np.load(directory/'initial.npz') as initial:
        np.testing.assert_allclose(trace['q_rad'][0], initial['qpos'][qpos], atol=1e-12, rtol=0)
        np.testing.assert_allclose(trace['qvel_rad_s'][0], initial['qvel'][dof], atol=1e-12, rtol=0)
        np.testing.assert_allclose(trace['sensor_command_rad'][0], initial['ctrl'][actuator_ids], atol=1e-12, rtol=0)
    dt = np.diff(trace['time_s'])
    assert np.all(dt > 0)
    speed = float(np.max(abs(np.diff(trace['command_rad'], axis=0))/dt[:, None])) if len(dt) else 0.
    assert speed <= 1.2+1e-8, speed
    return dict(fk_checked_states=len(fk_rows), maximum_fk_tip_error_m=max_fk,
                sensor_reconstructed_states=len(sensor_rows),
                maximum_reconstructed_force_error_n=float(errors[0]),
                maximum_reconstructed_contact_position_error_m=float(errors[1]),
                maximum_reconstructed_normal_component_error=float(errors[2]),
                reconstructed_contact_active_mismatches=active_errors,
                sensor_reconstruction_warmstart=('saved qacc_warmstart; saved q, qvel, prior control'
                    if 'qacc_warmstart' in trace else 'reset to zero; saved q, qvel, prior control'),
                unsafe_reconstructed_rows=unsafe_rows,
                arm_self_force_reconstruction_required=check_arm,
                arm_self_force_reconstructed_states=len(sensor_rows) if check_arm else 0,
                maximum_reconstructed_arm_self_force_error_n=arm_error if check_arm else None,
                maximum_sampled_arm_self_force_n=arm_peak if check_arm else None,
                unsafe_reconstructed_arm_self_rows=arm_rows,
                maximum_logged_command_speed_rad_s=speed,
                continuous_from_acquisition_initial_state=True)


def _safety_event_checks(directory, manifest, summary, trace, events):
    """Audit new native-tick event states without imposing v2.1 on old runs."""
    if not _arm_safety_required(manifest):
        return dict(v21_arm_safety_audited=False, native_force_event_states_checked=0)
    assert 'Module/i04_geometric_traversal/sliding_safety.py' in manifest['source_hashes']
    policy = manifest['arm_self_collision']
    assert policy['planning_clearance_m'] >= .001
    assert policy['physical_stop_force_n'] == .5
    assert 'arm_self_force_n' in trace and 'qacc_warmstart' in trace
    peak_arm = float(summary['peak_arm_self_force_n'])
    peak_finger = float(summary['peak_force_n'])
    assert np.isfinite(peak_arm) and peak_arm >= 0.
    assert np.isfinite(peak_finger) and peak_finger >= 0.
    assert np.max(trace['arm_self_force_n']) <= peak_arm+1e-9
    state_events = [(index, event) for index, event in enumerate(events)
                    if event['kind'] in ('FORCE_PEAK_STATE', 'HARDWARE_STOP')]
    context = _native_context(directory) if state_events else None
    checked, maximum_error, maximum_arm = [], 0., 0.
    peak_event_forces, shallow_stops = [], []
    for index, event in state_events:
        assert 0. <= event['time_s'] <= summary['executed_s']+1e-9
        for key in ('q_rad', 'qvel_rad_s', 'sensor_command_rad', 'qacc_warmstart', 'forces_n'):
            assert key in event, f'{event["kind"]} missing native state {key}'
        _restore_native(context, event)
        force, _, _, _, unsafe, _ = contact_measurement(context.handles, context.data, .2)
        recorded_force = np.asarray(event['forces_n'])
        assert recorded_force.shape == (4,) and np.all(np.isfinite(recorded_force))
        error = float(np.max(abs(force-recorded_force)))
        assert error <= 1e-5, (event['time_s'], error)
        contacts = _native_arm_contacts(context.model, context.data)
        arm_force = max([0.] + [contact['force_n'] for contact in contacts])
        maximum_error, maximum_arm = max(maximum_error, error), max(maximum_arm, arm_force)
        assert arm_force <= peak_arm+1e-5
        assert force.max() <= peak_finger+1e-5
        if event['kind'] == 'FORCE_PEAK_STATE':
            assert recorded_force.max() > 8.
            peak_event_forces.append(float(recorded_force.max()))
        else:
            logged_contacts = event['arm_self_contacts']
            assert len(logged_contacts) == len(contacts), event['time_s']
            for actual, logged in zip(contacts, logged_contacts):
                assert actual['geom_names'] == logged['geom_names']
                assert actual['body_names'] == logged['body_names']
                assert actual['contact_index'] == logged['contact_index']
                assert abs(actual['distance_m']-logged['distance_m']) <= 1e-9
                assert abs(actual['force_n']-logged['force_n']) <= 1e-5
            if event['reason'] == 'SHALLOW_ARM_SELF_COLLISION':
                assert arm_force > .5, 'Shallow self-collision stop has no physical force evidence'
                shallow_stops.append(event)
            elif arm_force > .5:
                raise AssertionError('Native arm self collision is missing its priority stop reason')
        checked.append(dict(source_event_index=index, kind=event['kind'], time_s=event['time_s'],
                            phase=event.get('phase'), reason=event.get('reason'),
                            recorded_finger_forces_n=recorded_force.tolist(),
                            reconstructed_finger_forces_n=force.tolist(),
                            maximum_finger_force_error_n=error,
                            maximum_arm_self_force_n=arm_force, arm_self_contacts=contacts,
                            other_unsafe_contacts=unsafe,
                            state_source='Original events.json q, qvel, prior control and warmstart'))
    assert all(b > a for a, b in zip(peak_event_forces, peak_event_forces[1:])), \
        'FORCE_PEAK_STATE events must record strictly increasing peaks'
    if peak_finger > 8.:
        assert peak_event_forces, 'Native-tick finger force peak has no saved state'
        assert abs(max(peak_event_forces)-peak_finger) <= 1e-9, 'Summary peak lacks a matching native state'
    else:
        assert not peak_event_forces
    if peak_arm > .5:
        assert shallow_stops, 'Physical arm-force threshold exceeded without a hardware stop'
        assert abs(maximum_arm-peak_arm) <= 1e-5, 'Arm-force peak is not reproduced by stop state'
    else:
        assert not shallow_stops
    hard_stops = [event for event in events if event['kind'] == 'HARDWARE_STOP']
    arm_limit_satisfied = peak_arm <= .5 and not shallow_stops
    if summary['full_surface_complete']:
        assert arm_limit_satisfied, 'Whole-surface completion includes an unsafe arm collision'
        assert not hard_stops, 'Whole-surface completion follows a hardware stop'
    event_digest = sha256((directory/'events.json').read_bytes()).hexdigest()
    report = dict(status='EVIDENCE_PASSED', scope='Native force-peak and stop-state replay',
                  events_sha256=event_digest, model_xml_sha256=sha256((directory/'model.xml').read_bytes()).hexdigest(),
                  state_count=len(checked), maximum_reconstructed_force_error_n=maximum_error,
                  summary_native_tick_peak_finger_force_n=peak_finger,
                  summary_native_tick_peak_arm_self_force_n=peak_arm,
                  maximum_sampled_trace_finger_force_n=float(trace['forces_n'].max()),
                  maximum_sampled_trace_arm_self_force_n=float(trace['arm_self_force_n'].max()),
                  force_sampling_note='10 ms trace can miss 2 ms force peaks; every new finger peak above 8 N has a replayed event',
                  arm_force_definition='Maximum individual contact normal force among nonadjacent FR3 links at distance <=0',
                  arm_self_collision_limit_satisfied=arm_limit_satisfied,
                  full_surface_complete=bool(summary['full_surface_complete']), states=checked)
    (directory/'force_event_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return dict(v21_arm_safety_audited=True,
                native_force_event_states_checked=len(checked),
                maximum_native_force_event_error_n=maximum_error,
                native_tick_peak_arm_self_force_n=peak_arm,
                native_tick_peak_finger_force_n=peak_finger,
                shallow_arm_self_collision_stop_count=len(shallow_stops),
                arm_self_collision_limit_satisfied=arm_limit_satisfied,
                force_event_audit_file='force_event_audit.json', events_sha256=event_digest)


def audit(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    with np.load(directory/'trace.npz') as archive:
        trace = {key: archive[key] for key in archive.files}
    assert len(trace['time_s']) > 0, 'No physical samples recorded'
    for name in ('time_s', 'q_rad', 'qvel_rad_s', 'command_rad', 'sensor_command_rad',
                 'forces_n', 'tip_positions_world_m', 'contact_positions_world_m', 'normals_world'):
        assert np.all(np.isfinite(trace[name])), name
    digest = sha256((directory/'trace.npz').read_bytes()).hexdigest()
    assert summary['trace_sha256'] == digest, 'Summary does not describe this trace'
    source = _source_checks(directory, manifest)
    coverage = _coverage(directory, manifest, summary, trace)
    physical = _physical_checks(directory, trace, manifest)
    finger_search = _active_finger_checks(directory, manifest, trace)
    events = json.loads((directory/'events.json').read_text(encoding='utf-8'))
    safety = _safety_event_checks(directory, manifest, summary, trace, events)
    normal_search = {}
    if manifest.get('protocol') == CONTEXT_PROTOCOLS[3]:
        from .normal_sliding_audit import normal_search_checks
        normal_search = normal_search_checks(directory, manifest, trace)
    relief_search = {}
    if manifest.get('protocol') in CONTEXT_PROTOCOLS[3:]:
        from .release_sliding_audit import applied_reference_relief_checks
        relief_search = applied_reference_relief_checks(directory, manifest, trace, events)
    expected = ['wrist_pose'] if manifest['cell'] == 'A-MCC' else ['wrist_pose', 'tip_positions']
    assert all(event['exposed_fields'] == expected for event in events if event['kind'] == 'SURFACE_STROKE')
    assert all(event['exposed_fields'] == expected for event in events if event['kind'] == 'NEAR_BRIDGE_COMMIT')
    lift_events = [event for event in events if event['kind'] == 'LIFT_COMMIT']
    # The first development pilot predates this event field. New logs expose
    # the transfer's target boundary explicitly and are checked as well.
    checked_lifts = [event for event in lift_events if 'exposed_fields' in event]
    assert all(event['exposed_fields'] == expected for event in checked_lifts)
    hard_stops = [event for event in events if event['kind'] == 'HARDWARE_STOP']
    if hard_stops:
        assert summary['stop_reason'] == hard_stops[-1]['reason']
        assert not summary['full_surface_complete']
    assert trace['time_s'][0] == 0. and trace['time_s'][-1] <= summary['executed_s']+1e-9
    assert float(trace['forces_n'].max()) <= summary['peak_force_n']+1e-9
    result = dict(status='PASSED', scope='Evidence integrity, not continuous-sliding success',
                  behavioral_acceptance='NOT_ASSESSED', safety_hard_stop=bool(hard_stops),
                  safety_hard_stop_events=hard_stops, stop_reason=summary['stop_reason'],
                  full_surface_complete=bool(summary['full_surface_complete']),
                  development_run=not manifest.get('full_surface_claim', False),
                  inherited_development_ledger=False, dp='DEFERRED_BY_USER',
                  lift_exposed_fields_checked=len(checked_lifts),
                  lift_events_without_exposed_fields=len(lift_events)-len(checked_lifts),
                  trace_sha256=digest, model_xml_sha256=sha256((directory/'model.xml').read_bytes()).hexdigest(),
                  summary_sha256=sha256((directory/'summary.json').read_bytes()).hexdigest(),
                  manifest_sha256=sha256((directory/'manifest.json').read_bytes()).hexdigest(),
                  auditor_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  **{**source, **coverage, **physical, **finger_search, **safety, **normal_search, **relief_search})
    (directory/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('SLIDING_AUDIT', json.dumps(result), flush=True)
    return result


def audit_batch(directory):
    """Aggregate the frozen legacy A/B six cases or active B-only three cases."""
    directory = Path(directory)
    available = [json.loads(path.read_text(encoding='utf-8')) for path in directory.glob('*/manifest.json')]
    active_only = bool(available and all(manifest.get('active_fingers', False) for manifest in available))
    cells = ('B-MCC',) if active_only else ('A-MCC', 'B-MCC')
    cases = [f'{cell}_seed{seed}' for seed in (7, 11, 19) for cell in cells]
    missing = [case for case in cases if not (directory/case/'summary.json').exists()]
    if missing:
        raise FileNotFoundError('Batch is incomplete: '+', '.join(missing))
    manifests = {case: json.loads((directory/case/'manifest.json').read_text(encoding='utf-8'))
                 for case in cases}
    expected_sources = manifests[cases[0]]['source_hashes']
    assert all(manifest['source_hashes'] == expected_sources for manifest in manifests.values()), \
        'Batch mixes runtime source versions'
    paired = []
    for seed in (() if active_only else (7, 11, 19)):
        with np.load(directory/f'A-MCC_seed{seed}'/'initial.npz') as a, \
                np.load(directory/f'B-MCC_seed{seed}'/'initial.npz') as b:
            for key in ('qpos', 'qvel', 'ctrl'):
                np.testing.assert_array_equal(a[key], b[key])
        paired.append(dict(seed=seed, qpos_exact=True, qvel_exact=True, ctrl_exact=True))
    results = []
    for case in cases:
        path = directory/case
        assert case == f"{manifests[case]['cell']}_seed{manifests[case]['seed']}", case
        _source_checks(path, manifests[case])
        prior = path/'audit.json'
        result = json.loads(prior.read_text(encoding='utf-8')) if prior.exists() else None
        digest = sha256((path/'trace.npz').read_bytes()).hexdigest()
        new_safety = _arm_safety_required(manifests[case])
        stale_safety = new_safety and (not result or not result.get('v21_arm_safety_audited')
            or result.get('events_sha256') != sha256((path/'events.json').read_bytes()).hexdigest()
            or result.get('summary_sha256') != sha256((path/'summary.json').read_bytes()).hexdigest()
            or result.get('manifest_sha256') != sha256((path/'manifest.json').read_bytes()).hexdigest()
            or result.get('auditor_sha256') != sha256(Path(__file__).read_bytes()).hexdigest())
        if not result or result.get('status') != 'PASSED' or result.get('trace_sha256') != digest or stale_safety:
            result = audit(path)
        assert result['status'] == 'PASSED'
        assert result['model_xml_sha256'] == sha256((path/'model.xml').read_bytes()).hexdigest(), case
        assert result['lift_events_without_exposed_fields'] == 0, case
        results.append(dict(case=case, stop_reason=result['stop_reason'],
                            safety_hard_stop=result['safety_hard_stop'],
                            full_surface_complete=result['full_surface_complete'],
                            area_fraction=result['area_fraction'], vertices=result['vertices'],
                            required_goals=result['required_goals'], trace_sha256=digest,
                            maximum_fk_tip_error_m=result['maximum_fk_tip_error_m'],
                            maximum_reconstructed_force_error_n=result['maximum_reconstructed_force_error_n'],
                            v21_arm_safety_audited=result.get('v21_arm_safety_audited', False),
                            arm_self_collision_limit_satisfied=result.get('arm_self_collision_limit_satisfied'),
                            native_tick_peak_arm_self_force_n=result.get('native_tick_peak_arm_self_force_n')))
    aggregate = dict(status='EVIDENCE_PASSED', scope='Recorded traces; behavioral success not asserted',
                     controller_information_scope=('B-only geometry-guided active finger planning'
                        if active_only else 'Legacy matched A/B target-information comparison'),
                     completed_case_count=len(cases), passed_evidence_audits=len(cases),
                     safety_hard_stop_count=sum(result['safety_hard_stop'] for result in results),
                     full_surface_complete_count=sum(result['full_surface_complete'] for result in results),
                     behavioral_acceptance='NOT_ASSESSED',
                     same_runtime_sources=True, runtime_source_count=len(expected_sources),
                     source_hashes=expected_sources, paired_initial_states=paired,
                     all_cases_total_vertices=8192, all_cases_total_required_goals=388,
                     coverage_radius_m=.025, cases=results)
    (directory/'audit_summary.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    print('SLIDING_BATCH_AUDIT', json.dumps(aggregate), flush=True)
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--batch', action='store_true')
    args = parser.parse_args()
    (audit_batch if args.batch else audit)(args.directory)


if __name__ == '__main__':
    main()
