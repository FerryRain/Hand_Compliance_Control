"""Independent continuous native replay of saved-state posture recovery probes."""
from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath

import mujoco
import numpy as np

from Module.e05_physics.scene import FINGERS
from .physics import contact_measurement
from .sliding_audit import _native_context, _native_arm_contacts


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def _runs(mask, times):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a), int(b)-1, float(times[b-1]-times[a]))
            for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def audit(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    assert manifest['protocol'] in ('i04-anchored-posture-recovery.paired.development.v1',
                                    'i04-anchored-posture-recovery.paired.development.v2',
                                    'i04-anchored-posture-recovery.paired.development.v3',
                                    'i04-anchored-posture-recovery.paired.development.v4')
    new_handoff = manifest['protocol'].endswith(('.v2', '.v3', '.v4'))
    persistent = manifest.get('persistent_search',False)
    assert manifest['coverage_authority'] == 'NONE' and not manifest['coverage_counted']
    assert not summary['whole_traversal_evidence'] and not summary['coverage_counted']
    assert _hash(directory/'trace.npz') == summary['trace_sha256']
    assert _hash(directory/'model.xml') == manifest['model_sha256']
    for name, expected in manifest['source_hashes'].items():
        relative = PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert _hash(directory/'source_snapshot'/relative) == expected, name
    runtime = directory/'source_snapshot/Module/i04_geometric_traversal/posture_recovery_probe.py'
    tree = ast.parse(runtime.read_text(encoding='utf-8'))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    for node in ast.walk(run):
        if isinstance(node, ast.Call):
            assert not (isinstance(node.func, ast.Name) and node.func.id == 'set_joint_state')
        targets = (node.targets if isinstance(node, ast.Assign) else [node.target]
                   if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
        assert not any(isinstance(item, ast.Attribute) and item.attr == 'qpos'
                       for target in targets for item in ast.walk(target))
    provenance = manifest['source_state']
    assert _hash(directory/'source_run_manifest.json') == provenance['source_manifest_sha256']
    assert _hash(directory/'surface_graph.npz') == provenance['source_graph_sha256']
    assert _hash(directory/'source_clock_anchor.npz') == provenance['clock_anchor_sha256']
    anchor_state = _load(directory/'source_clock_anchor.npz')
    clock = 0.
    source_native = None
    source_step = provenance['acquisition_steps']+round(provenance['source_trace']['relative_time_s']/.002)
    anchor_step = provenance['acquisition_steps']+round(float(anchor_state['elapsed'])/.002)
    for step in range(anchor_step+1):
        if step == source_step:
            source_native = clock
        if step < anchor_step:
            clock += .002
    assert clock == float(anchor_state['time']) and source_native == provenance['native_time_s']
    source_frame = _load(directory/'source_frame.npz')
    origin = provenance['source_trace']
    assert _hash(origin['path']) == origin['sha256']
    with np.load(origin['path']) as original:
        for key, value in source_frame.items():
            np.testing.assert_array_equal(value, original[key][origin['row']])
    tr, initial = _load(directory/'trace.npz'), _load(directory/'initial.npz')
    context = _native_context(directory)
    model, data, h = context.model, context.data, context.handles
    assert model.opt.timestep == .002 and model.na == 0
    np.testing.assert_allclose(np.diff(tr['time_s']), .002, atol=1e-12, rtol=0)
    assert len(tr['time_s']) == summary['native_recorded_states']
    assert tr['time_s'][0] == 0. and tr['time_s'][-1] == summary['executed_s']
    for key in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
        getattr(data, key)[:] = initial[key]
    data.time = float(initial['time'])
    assert data.time == source_native
    np.testing.assert_array_equal(data.qpos[context.qpos], source_frame['q_rad'])
    np.testing.assert_array_equal(data.qvel[context.dof], source_frame['qvel_rad_s'])
    np.testing.assert_array_equal(data.ctrl[context.actuator_ids], source_frame['sensor_command_rad'])
    np.testing.assert_array_equal(data.qacc_warmstart, source_frame['qacc_warmstart'])
    if new_handoff:
        assert _hash(directory/'controller_handoff.npz') == manifest['controller_handoff_sha256']
        handoff = _load(directory/'controller_handoff.npz')
        np.testing.assert_array_equal(handoff['raw_path_root_q'], source_frame['q_rad'])
        np.testing.assert_array_equal(handoff['prior_control_rad'], source_frame['sensor_command_rad'])
        np.testing.assert_array_equal(handoff['support_preload_delta_rad'],
                                      source_frame['sensor_command_rad']-source_frame['q_rad'])
        others = handoff['unselected_hand_columns']
        np.testing.assert_array_equal(handoff['applied_initial_reference_rad'][others],
                                      source_frame['sensor_command_rad'][others])
    mujoco.mj_forward(model, data)
    maximum = {key: 0. for key in ('q_rad', 'qvel_rad_s', 'prior_control_rad', 'warmstart',
        'native_time_s', 'force_n', 'point_m', 'normal', 'tip_fk_m', 'arm_self_force_n')}
    active_errors, unsafe_frames, arm_exceed_frames = 0, [], []
    peak_force, peak_arm = 0., 0.
    for row in range(len(tr['time_s'])):
        # Exactly one initialization; all subsequent states are obtained by
        # replaying recorded controls through native 2 ms integration.
        pairs = dict(q_rad=(data.qpos[context.qpos], tr['q_rad'][row]),
            qvel_rad_s=(data.qvel[context.dof], tr['qvel_rad_s'][row]),
            prior_control_rad=(data.ctrl[context.actuator_ids], tr['sensor_command_rad'][row]),
            warmstart=(data.qacc_warmstart, tr['qacc_warmstart'][row]),
            native_time_s=(data.time, tr['native_time_s'][row]),
            tip_fk_m=(data.site_xpos[context.sites], tr['tip_positions_world_m'][row]))
        forces, points, normals, active, unsafe, _ = contact_measurement(h, data, .2)
        arm_force = max([0.] + [c['force_n'] for c in _native_arm_contacts(model, data)])
        pairs.update(force_n=(forces, tr['forces_n'][row]), point_m=(points, tr['contact_positions_world_m'][row]),
                     normal=(normals, tr['normals_world'][row]), arm_self_force_n=(arm_force, tr['arm_self_force_n'][row]))
        for key, (actual, recorded) in pairs.items():
            maximum[key] = max(maximum[key], float(np.max(abs(np.asarray(actual)-recorded))))
        active_errors += int(np.count_nonzero(active != tr['active'][row]))
        if unsafe:
            unsafe_frames.append(row)
        if arm_force > .5:
            arm_exceed_frames.append(row)
        peak_force, peak_arm = max(peak_force, float(forces.max())), max(peak_arm, arm_force)
        if row+1 < len(tr['time_s']):
            data.ctrl[context.actuator_ids] = tr['command_rad'][row]
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
    assert maximum['q_rad'] < 1e-8 and maximum['qvel_rad_s'] < 1e-7, maximum
    assert maximum['force_n'] < 1e-5 and maximum['arm_self_force_n'] < 1e-5, maximum
    assert all(maximum[k] < 1e-7 for k in ('prior_control_rad', 'point_m', 'normal', 'tip_fk_m')), maximum
    assert maximum['native_time_s'] == 0. and maximum['warmstart'] < 1e-7, maximum
    assert active_errors == 0
    assert abs(peak_force-summary['peak_force_n']) < 1e-5 and abs(peak_arm-summary['peak_arm_self_force_n']) < 1e-5
    final = _load(directory/'final_state.npz')
    for key in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
        np.testing.assert_allclose(getattr(data, key), final[key], atol=1e-7, rtol=0)
    assert data.time == final['time']
    command_step = float(np.max(abs(tr['command_rad']-tr['sensor_command_rad'])))
    assert command_step <= .0024+1e-10, command_step
    finger, anchor = int(manifest['selected_finger']), int(manifest['anchor_finger'])
    selected_columns = np.asarray([int(name) for name in FINGERS[finger].joint_names])+7
    other_columns = np.setdiff1d(np.arange(23), selected_columns)
    if 'selected_search_reference_rad' in tr:
        np.testing.assert_array_equal(tr['selected_search_reference_rad'][:, other_columns],
                                      tr['finger_search_nominal_rad' if persistent else 'nominal_reference_rad'][:, other_columns])
    assert not np.any(tr['finger_search_active'][:, np.arange(4) != finger])
    assert not np.any(tr['finger_search_active'] & tr['active'])
    assert not np.any(tr['coverage']) and not np.any(tr['covered_goals'])
    if not manifest['posture_enabled'] and 'nominal_reference_rad' in tr:
        np.testing.assert_array_equal(tr['nominal_reference_rad'][:, :7],
            np.tile(tr['q_rad'][0, :7], (len(tr['time_s']), 1)))
    safe = tr['active'][:, finger] & (tr['forces_n'][:, finger] >= .2) & (tr['forces_n'][:, finger] <= 8.)
    safe_anchor = tr['active'][:, anchor] & (tr['forces_n'][:, anchor] >= .2) & (tr['forces_n'][:, anchor] <= 8.)
    stable_runs = [r for r in _runs(safe, tr['time_s']) if r[2] >= .15-1e-10]
    confirmation = float(tr['time_s'][stable_runs[0][0]]+.15) if stable_runs else None
    first_touch = float(tr['time_s'][np.flatnonzero(safe)[0]]) if np.any(safe) else None
    assert bool(stable_runs) == summary['stable_recontact_observed']
    if confirmation is None:
        assert summary['stable_confirmation_delay_s'] is None
    else:
        assert abs(confirmation-summary['stable_confirmation_delay_s']) < 1e-10
    assert first_touch == summary['first_recontact_delay_s']
    complete = summary['stop_reason'] == 'RECOVERY_WINDOW_COMPLETE'
    assert summary['success'] == bool(stable_runs and complete)
    if complete:
        assert abs(tr['time_s'][-1]-manifest['observation_duration_s']) < 1e-10
        assert not unsafe_frames and not arm_exceed_frames
    events = json.loads((directory/'events.json').read_text(encoding='utf-8'))
    combined_evidence = None
    if manifest.get('recovery_mode') == 'pivot_explicit_target':
        spec = manifest['combined_strategy']
        assert manifest['search_implementation'] == ('ActiveFingerSearchV3' if persistent else 'BoundedOnlyFingerSearch')
        assert not spec['stage2_state_reset'] and not spec['pivot_selected_search_enabled']
        mesh = _load(directory/'surface_graph.npz')
        target = mesh['vertices'][mesh['faces'][spec['target_mesh_face_id']]].mean(axis=0)
        np.testing.assert_allclose(target, spec['target_point_local_m'], atol=1e-14, rtol=0)
        for name, expected in summary['dynamic_artifact_hashes'].items():
            assert name in ('stage2_actual_root.npz', 'stage2_planning.json', 'explicit_target_path.npz')
            assert _hash(directory/name) == expected
        replans = [e for e in events if e['kind'] == 'ACTUAL_PIVOT_ENDPOINT_REPLAN']
        assert len(replans) <= 1
        if not manifest['intervention_enabled']:
            assert not replans and not summary['dynamic_artifact_hashes']
            assert not np.any(tr['posture_path_elapsed_s']) and not np.any(tr['explicit_target_path_elapsed_s'])
        pivot = np.isin(tr['combined_stage'], ['PIVOT', 'PIVOT_SETTLE'])
        assert not np.any(tr['finger_search_enabled'][pivot])
        assert not np.any(tr['finger_search_active'][pivot])
        pivot_path = _load(directory/'posture_path.npz')
        pivot_duration = float(np.sum(pivot_path['segment_durations']))
        for row in np.flatnonzero((tr['combined_stage'] == 'PIVOT')[:-1] & ~safe_anchor[:-1]):
            assert tr['posture_path_elapsed_s'][row+1] == tr['posture_path_elapsed_s'][row]
        replan_time = None
        if replans:
            e = replans[0]
            row = int(np.flatnonzero(abs(tr['time_s']-e['time_s']) < 1e-12)[0])
            replan_time = float(e['time_s'])
            assert {'stage2_actual_root.npz', 'stage2_planning.json'} <= summary['dynamic_artifact_hashes'].keys()
            root = _load(directory/'stage2_actual_root.npz')
            for event_key, trace_key in (('q_rad','q_rad'), ('qvel_rad_s','qvel_rad_s'),
                    ('sensor_command_rad','sensor_command_rad'), ('qacc_warmstart','qacc_warmstart')):
                np.testing.assert_array_equal(e[event_key], tr[trace_key][row])
            assert e['native_time_s'] == tr['native_time_s'][row] == float(root['time'])
            for key, index, trace_key in (('qpos',context.qpos,'q_rad'),
                    ('qvel',context.dof,'qvel_rad_s'),('ctrl',context.actuator_ids,'sensor_command_rad')):
                np.testing.assert_array_equal(root[key][index],tr[trace_key][row])
            np.testing.assert_array_equal(root['qacc_warmstart'],tr['qacc_warmstart'][row])
            np.testing.assert_array_equal(root['q_rad'],tr['q_rad'][row])
            np.testing.assert_array_equal(root['raw_pivot_endpoint_q'],pivot_path['q'][-1])
            np.testing.assert_array_equal(root['initial_reference_rad'][:7],tr['q_rad'][row,:7])
            np.testing.assert_array_equal(root['initial_reference_rad'][others],tr['sensor_command_rad'][row,others])
            assert tr['posture_path_elapsed_s'][row] >= pivot_duration-1e-12
            reached = np.flatnonzero(tr['posture_path_elapsed_s'] >= pivot_duration-1e-12)[0]
            assert e['time_s']-tr['time_s'][reached] >= spec['settle_minimum_s']-1e-10
            stable_window = (tr['time_s'] >= e['time_s']-spec['settled_tracking_confirmation_s']-1e-12) & (tr['time_s'] <= e['time_s'])
            endpoint = pivot_path['q'][-1,:7]
            for a,b in ((tr['q_rad'][stable_window,:7],endpoint),
                        (tr['q_rad'][stable_window,:7],tr['sensor_command_rad'][stable_window,:7]),
                        (tr['sensor_command_rad'][stable_window,:7],endpoint)):
                assert np.max(abs(a-b)) <= spec['settled_tracking_threshold_rad']+1e-10
            actual_safe = tr['active'][stable_window] & (tr['forces_n'][stable_window] >= .2) & (tr['forces_n'][stable_window] <= 8.)
            assert np.all(np.any(actual_safe,axis=1))
            assert np.max(tr['forces_n'][stable_window]) < 8.
            assert e['native_plant_unchanged']
            if 'explicit_target_path.npz' in summary['dynamic_artifact_hashes']:
                target_path = _load(directory/'explicit_target_path.npz')
                np.testing.assert_array_equal(target_path['q'][0],tr['q_rad'][row])
                other = np.setdiff1d(np.arange(23),selected_columns)
                np.testing.assert_array_equal(target_path['q'][:,other],
                    np.tile(tr['q_rad'][row,other],(len(target_path['q']),1)))
            if e['accepted']:
                assert summary['explicit_target_path_available']
                assert 'explicit_target_path.npz' in summary['dynamic_artifact_hashes']
            if np.any(safe) and not summary.get('reference_replans',0):
                touch_row = int(np.flatnonzero(safe)[0])
                assert np.ptp(tr['explicit_target_path_elapsed_s'][touch_row:]) == 0.
                assert np.ptp(tr['posture_path_elapsed_s'][touch_row:]) == 0.
        combined_evidence = dict(original_mesh_target_verified=True, selected_search_masked_during_pivot=True,
            actual_endpoint_replan_count=len(replans), actual_endpoint_replan_time_s=replan_time,
            dynamic_artifact_hashes_verified=len(summary['dynamic_artifact_hashes']),
            endpoint_replan_uses_actual_native_state=True if replans else None)
        if persistent:
            assert manifest['mcc_implementation']=='ContactPreservingMCC'
            rebase_events=[e for e in events if e['kind']=='REFERENCE_REBASE_ACTUAL_REPLAN']
            assert len(rebase_events)==summary['reference_replans']
            required=np.flatnonzero(tr['finger_search_requires_rebase'][:,finger])
            assert len(required)==len(rebase_events)
            for name,expected in summary['reference_replan_artifact_hashes'].items():
                relative=PurePosixPath(name)
                assert not relative.is_absolute() and '..' not in relative.parts and relative.parts[0]=='reference_replans'
                assert _hash(directory/relative)==expected
            for row,e in zip(required,rebase_events):
                assert e['time_s']==tr['time_s'][row]
                for key in ('q_rad','qvel_rad_s','sensor_command_rad','qacc_warmstart'):
                    np.testing.assert_array_equal(e[key],tr[key][row])
                assert e['native_time_s']==tr['native_time_s'][row]
                folder=directory/'reference_replans'/f"{e['index']:02d}"
                root=_load(folder/'actual_root.npz')
                np.testing.assert_array_equal(root['qpos'][context.qpos],tr['q_rad'][row])
                np.testing.assert_array_equal(root['qvel'][context.dof],tr['qvel_rad_s'][row])
                np.testing.assert_array_equal(root['ctrl'][context.actuator_ids],tr['sensor_command_rad'][row])
                np.testing.assert_array_equal(root['qacc_warmstart'],tr['qacc_warmstart'][row])
                if e['accepted']:
                    assert e['grant']['accepted']
                    np.testing.assert_array_equal(_load(folder/'path.npz')['q'][0],tr['q_rad'][row])
            seeking=tr['finger_search_active'][:,finger]
            assert np.all(tr['finger_search_elapsed_s'][seeking,finger]<manifest['active_finger_search']['maximum_search_time_s']+1e-10)
            assert np.all(tr['finger_search_loss_age_s'][seeking,finger]>=manifest['active_finger_search']['loss_debounce_s']-1e-10)
            assert np.all(tr['finger_posture_attempts_this_episode']<=3)
            assert np.all(np.diff(tr['finger_posture_attempt_id'],axis=0)>=0)
            combined_evidence.update(reference_replans_verified=len(rebase_events),persistent_search_lifecycle_checked=True)
    if new_handoff:
        handoff_events = [e for e in events if e['kind'] == 'SELECTED_CONTACT_REFERENCE_HANDOFF']
        assert len(handoff_events) == int(np.any(safe)) if not persistent else len(handoff_events)>=int(np.any(safe))
        if handoff_events:
            assert handoff_events[0]['time_s']==first_touch
            for e in handoff_events:
                row=int(np.flatnonzero(abs(tr['time_s']-e['time_s'])<1e-12)[0])
                assert safe[row]
                np.testing.assert_array_equal(e['preserved_prior_control_rad'], tr['sensor_command_rad'][row, selected_columns])
                np.testing.assert_allclose(tr['finger_search_nominal_rad' if persistent else 'nominal_reference_rad'][row, selected_columns],
                    tr['sensor_command_rad'][row, selected_columns],atol=1e-12,rtol=0)
    if arm_exceed_frames or unsafe_frames:
        assert any(e['kind'] == 'HARDWARE_STOP' for e in events)
        assert set(arm_exceed_frames+unsafe_frames) == {len(tr['time_s'])-1}
    result = dict(status='PASSED', scope='Evidence integrity for this saved-state probe; no whole-surface coverage',
        protocol=manifest['protocol'], source_snapshots_verified=len(manifest['source_hashes']),
        source_state_verified=True, native_clock_exact=True, replay_initializations=1,
        replay_state_restores_after_initialization=0, source_direct_joint_teleports=0,
        native_states_replayed=len(tr['time_s']), maximum_native_replay_errors=maximum,
        actual_contact_mismatches=active_errors, selected_finger=finger, anchor_finger=anchor,
        same_finger_stable_recontact_observed=bool(stable_runs), first_recontact_delay_s=first_touch,
        stable_confirmation_delay_s=confirmation, complete_observation_window=complete,
        stop_reason=summary['stop_reason'], successful_complete_recovery=bool(stable_runs and complete),
        peak_force_n=peak_force, peak_actual_arm_self_force_n=peak_arm,
        unsafe_physical_frames=unsafe_frames, arm_self_above_threshold_frames=arm_exceed_frames,
        maximum_command_step_rad=command_step, search_applied_only_to_selected_reference=True,
        safe_anchor_sample_fraction=float(np.mean(safe_anchor)),
        maximum_anchor_loss_s=max([0.]+[float(tr['time_s'][min(r[1]+1,len(safe_anchor)-1)]-tr['time_s'][r[0]])
            for r in _runs(~safe_anchor, tr['time_s'])]),
        selected_safe_sample_fraction=float(np.mean(safe)),
        recovery_mode=manifest.get('recovery_mode', 'wrist'),
        combined_strategy_evidence=combined_evidence,
        audit_source_sha256=_hash(__file__), mujoco_version=mujoco.__version__,
        coverage_counted=False, trace_sha256=summary['trace_sha256'])
    (directory/'native_probe_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('POSTURE_PROBE_AUDIT', json.dumps(result), flush=True)
    return result


def audit_pair(directory):
    directory = Path(directory)
    manifests = {case: json.loads((directory/case/'manifest.json').read_text(encoding='utf-8')) for case in ('ON', 'OFF')}
    initial = {case: _load(directory/case/'initial.npz') for case in manifests}
    for key in initial['ON']:
        np.testing.assert_array_equal(initial['ON'][key], initial['OFF'][key])
    assert manifests['ON'].get('intervention_enabled', manifests['ON']['posture_enabled'])
    assert not manifests['OFF'].get('intervention_enabled', manifests['OFF']['posture_enabled'])
    for key in ('source_hashes', 'source_state', 'model_sha256', 'common_controls',
                'active_finger_search', 'posture_planner', 'observation_duration_s', 'geometric_path_available'):
        assert manifests['ON'][key] == manifests['OFF'][key], key
    if manifests['ON']['geometric_path_available']:
        paths = {case: _load(directory/case/'posture_path.npz') for case in manifests}
        for key in paths['ON']:
            np.testing.assert_array_equal(paths['ON'][key], paths['OFF'][key])
    results = {case: audit(directory/case) for case in manifests}
    mode = manifests['ON'].get('recovery_mode', 'wrist')
    difference = dict(wrist='ON applies the geometric anchored wrist proposal',
        bounded_only='ON replaces only the local DLS-then-clip solver with the bounded solver',
        finger_reconfiguration='ON replaces local search with a full safe selected-finger path until first contact',
        pivot_explicit_target='ON adds an anchored pivot, actual endpoint replan to an original mesh target, and selected-finger path; both use bounded local search')[mode]
    if mode == 'pivot_explicit_target':
        assert manifests['ON']['combined_strategy'] == manifests['OFF']['combined_strategy']
        expected='ActiveFingerSearchV3' if manifests['ON'].get('persistent_search') else 'BoundedOnlyFingerSearch'
        assert manifests['ON']['search_implementation'] == manifests['OFF']['search_implementation'] == expected
    if mode == 'bounded_only':
        assert manifests['ON']['search_implementation'] == 'BoundedOnlyFingerSearch'
        assert manifests['OFF']['search_implementation'] == 'ActiveFingerSearch'
        assert not manifests['ON']['posture_enabled'] and not manifests['OFF']['posture_enabled']
    result = dict(status='PASSED', identical_initial_state=True, identical_runtime_sources=True,
        identical_common_controller_configuration=True, difference=difference,
        scope='Paired saved-state recovery; coverage remains zero', cases=results)
    (directory/'paired_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--single', action='store_true')
    args = parser.parse_args()
    (audit if args.single else audit_pair)(args.directory)


if __name__ == '__main__':
    main()
