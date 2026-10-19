"""Native paired recovery strategies from one frozen traversal state.

Both cases start with exactly the same q, qvel, prior control, solver warmstart
and verified native clock. Explicit modes compare an anchored wrist path,
bounded local search, or full selected-finger reconfiguration to their stated
baseline. Only the selected missing finger receives search/reconfiguration.
Other fingers retain prior-control preload. Actual contacts receive MCC.
This probe never awards whole-surface coverage or resets the moving plant.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import mujoco
import numpy as np
import trimesh

from Module.e05_physics.scene import FINGERS
from Module.fr3_leap import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from Module.fr3_leap.model import compile_model_xml
from .active_fingers import ActiveFingerSearch
from .contact_recovery_planner import ContactRecoveryPlanner
from .extended_contact_recovery import ExtendedContactRecoveryConfig
from .explicit_surface_finger_planner import ExplicitSurfaceFingerPlanner
from .finger_reconfiguration_planner import FingerReconfigurationPlanner
from .active_fingers_bounded_only import BoundedOnlyFingerSearch
from .active_fingers_v3 import ActiveFingerSearchV3
from .contact_handoff import ContactPreservingMCC
from .contracts import ExperimentConfig
from .fixtures import SurfaceGraph
from .physics import (BasicPassiveMCC, contact_measurement, joint_state, pose,
                      set_control, tip_positions)
from .recovery import SharedContactRecovery
from .sliding_safety import SlidingSafetyPlanner, arm_self_contacts


DT = .002
STATE_FIELDS = ('q_rad', 'qvel_rad_s', 'sensor_command_rad', 'qacc_warmstart')


def _digest(data):
    return sha256(data).hexdigest()


def _native_time(steps):
    value = 0.
    for _ in range(int(steps)):
        value += DT
    return value


def prepare_source(directory, time_s, selected_finger):
    """Read an immutable trace row and verify its absolute integration clock."""
    directory = Path(directory).resolve()
    if selected_finger not in range(4) or not np.isfinite(time_s) or time_s < 0:
        raise ValueError('Expected a nonnegative source time and a finger 0..3')
    manifest_bytes = (directory/'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    for name, expected in manifest['source_hashes'].items():
        assert _digest((directory/'source_snapshot'/name).read_bytes()) == expected, name
    chosen = None
    paths = ([directory/'trace.npz'] if (directory/'trace.npz').exists()
             else sorted((directory/'trace_chunks').glob('*.npz')))
    for path in paths:
        raw = path.read_bytes()
        with np.load(io.BytesIO(raw)) as archive:
            found = np.flatnonzero(abs(archive['time_s']-time_s) < 1e-10)
            if len(found):
                row = int(found[0])
                fields = STATE_FIELDS + ('forces_n', 'normals_world', 'active',
                    'contact_positions_world_m', 'tip_positions_world_m', 'palm_pose_world')
                chosen = {key: archive[key][row].copy() for key in fields}
                origin = dict(path=str(path), sha256=_digest(raw), row=row,
                              relative_time_s=float(archive['time_s'][row]))
                break
    if chosen is None:
        raise ValueError(f'No saved native state at {time_s:g} s')
    if bool(chosen['active'][selected_finger]):
        raise ValueError('The selected finger must actually be missing at the source state')
    # Whole-surface traces historically saved only relative time. The native
    # checkpoint clock verifies exact repeated double additions, including the
    # 5 s acquisition. Never substitute time_s + 5, which rounds differently.
    anchor_bytes = (directory/'coverage_checkpoint.npz').read_bytes()
    with np.load(io.BytesIO(anchor_bytes)) as anchor:
        elapsed, clock = float(anchor['elapsed']), float(anchor['time'])
    acquisition_steps = int(round((clock-elapsed)/DT))
    assert acquisition_steps >= 0 and abs(elapsed/DT-round(elapsed/DT)) < 1e-6
    reconstructed = _native_time(acquisition_steps+round(elapsed/DT))
    assert reconstructed == clock, 'Native clock cannot be reproduced exactly'
    assert abs(time_s/DT-round(time_s/DT)) < 1e-6
    native_time = _native_time(acquisition_steps+round(time_s/DT))
    model_bytes = (directory/'model.xml').read_bytes()
    graph_bytes = (directory/'surface_graph.npz').read_bytes()
    provenance = dict(source_run=str(directory), source_manifest_sha256=_digest(manifest_bytes),
        source_model_sha256=_digest(model_bytes), source_graph_sha256=_digest(graph_bytes),
        source_trace=origin, selected_finger=selected_finger,
        native_time_s=native_time, clock_anchor_elapsed_s=elapsed, clock_anchor_time_s=clock,
        clock_anchor_sha256=_digest(anchor_bytes), acquisition_steps=acquisition_steps,
        native_clock_reconstruction_exact=True,
        source_runtime_hashes=manifest['source_hashes'])
    return dict(frame=chosen, model_bytes=model_bytes, graph_bytes=graph_bytes,
                source_manifest_bytes=manifest_bytes, clock_anchor_bytes=anchor_bytes,
                object_position_m=manifest['object_position_m'], seed=int(manifest['seed']),
                provenance=provenance)


def make_plant(source):
    """Initialize a fresh physical replica once; no later state restoration."""
    model = compile_model_xml(source['model_bytes'].decode('utf-8'))
    assert (model.nq, model.nv, model.nu, model.na) == (23, 23, 23, 0)
    assert model.opt.timestep == DT
    def ids(kind, names):
        values = np.array([mujoco.mj_name2id(model, kind, name) for name in names])
        assert np.all(values >= 0), names
        return values
    joints = ids(mujoco.mjtObj.mjOBJ_JOINT, (*ARM_JOINT_NAMES, *HAND_JOINT_NAMES))
    actuators = ids(mujoco.mjtObj.mjOBJ_ACTUATOR,
        (*[f'fr3_arm_act_{i}' for i in range(1, 8)], *[f'act_{i}' for i in range(16)]))
    qpos, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    sites = ids(mujoco.mjtObj.mjOBJ_SITE, [f.pad_site.replace('e05_', 'fr3_') for f in FINGERS])
    geoms = ids(mujoco.mjtObj.mjOBJ_GEOM, [f.proxy_geom.replace('e05_', 'fr3_') for f in FINGERS])
    body = int(ids(mujoco.mjtObj.mjOBJ_BODY, ['fr3_e05_object'])[0])
    objects = np.flatnonzero(model.geom_bodyid == body)
    h = SimpleNamespace(model=model, xml=source['model_bytes'].decode('utf-8'),
        arm_qpos_adrs=qpos[:7], hand_qpos_adrs=qpos[7:], arm_dof_adrs=dofs[:7], hand_dof_adrs=dofs[7:],
        arm_actuator_ids=actuators[:7], hand_actuator_ids=actuators[7:],
        arm_joint_ranges_rad=model.jnt_range[joints[:7]].copy(),
        hand_joint_ranges_rad=model.jnt_range[joints[7:]].copy(),
        tip_geom_ids=geoms, tip_site_ids=sites, object_geom_ids=objects, object_geom_id=int(objects[0]),
        palm_site_id=int(ids(mujoco.mjtObj.mjOBJ_SITE, ['fr3_palm_control_site'])[0]),
        object_position_m=np.array(source['object_position_m']))
    frame = source['frame']
    data = mujoco.MjData(model)
    data.qpos[qpos] = frame['q_rad']
    data.qvel[dofs] = frame['qvel_rad_s']
    data.ctrl[actuators] = frame['sensor_command_rad']
    data.qacc_warmstart[:] = frame['qacc_warmstart']
    data.time = source['provenance']['native_time_s']
    mujoco.mj_forward(model, data)
    forces, points, normals, active, unsafe, _ = contact_measurement(h, data, .2)
    errors = dict(force_n=float(np.max(abs(forces-frame['forces_n']))),
        point_m=float(np.max(abs(points-frame['contact_positions_world_m']))),
        normal=float(np.max(abs(normals-frame['normals_world']))),
        tip_fk_m=float(np.max(abs(tip_positions(h, data)-frame['tip_positions_world_m']))))
    assert errors['force_n'] <= 1e-5 and max(errors[k] for k in ('point_m', 'normal', 'tip_fk_m')) <= 1e-7
    np.testing.assert_array_equal(active, frame['active'])
    assert not unsafe and not arm_self_contacts(h, data), 'Unsafe source condition'
    with np.load(io.BytesIO(source['graph_bytes'])) as graph_data:
        mesh = trimesh.Trimesh(vertices=graph_data['vertices'], faces=graph_data['faces'], process=False)
        graph = SurfaceGraph(mesh, coverage_radius_m=.025)
        np.testing.assert_array_equal(graph.vertices_m, graph_data['vertices'])
        np.testing.assert_array_equal(graph.required_vertices, graph_data['required_vertices'])
    return h, graph, data, errors


def snapshot_sources():
    """Freeze all imported repository Python modules, including both new files."""
    root = Path(__file__).resolve().parents[2]
    paths = {Path(__file__).resolve()}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if name.startswith('Module.') and filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(root):
                paths.add(path)
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(paths)}


def _state(h, data):
    return dict(q_rad=joint_state(h, data).tolist(),
        qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs], data.qvel[h.hand_dof_adrs]].tolist(),
        sensor_command_rad=np.r_[data.ctrl[h.arm_actuator_ids], data.ctrl[h.hand_actuator_ids]].tolist(),
        qacc_warmstart=data.qacc_warmstart.tolist(), native_time_s=float(data.time))


def run(directory, source, *, posture_enabled=True, duration_s=25., sources=None,
        recovery_mode='wrist', persistent_search=False):
    if not np.isfinite(duration_s) or duration_s < .15:
        raise ValueError('Observation duration must be at least 150 ms')
    if recovery_mode not in ('wrist', 'bounded_only', 'finger_reconfiguration', 'pivot_explicit_target'):
        raise ValueError('Unsupported recovery comparison')
    intervention_enabled = bool(posture_enabled)
    combined = recovery_mode == 'pivot_explicit_target'
    if persistent_search and not combined:
        raise ValueError('Persistent-search comparison is explicitly scoped to the combined probe')
    wrist_enabled = intervention_enabled and recovery_mode in ('wrist', 'pivot_explicit_target')
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Use an empty new probe directory')
    directory.mkdir(parents=True, exist_ok=True)
    sources = snapshot_sources() if sources is None else sources
    started = perf_counter()
    h, graph, data, sensor_errors = make_plant(source)
    finger = int(source['provenance']['selected_finger'])
    cfg = ExperimentConfig(seed=source['seed'])
    guard = SlidingSafetyPlanner(h, cfg)
    search_type = ActiveFingerSearchV3 if persistent_search else BoundedOnlyFingerSearch if combined or (recovery_mode == 'bounded_only' and intervention_enabled) else ActiveFingerSearch
    search = search_type(h, graph)
    mcc = ContactPreservingMCC(h, DT) if persistent_search else BasicPassiveMCC(h, DT)
    relief = SharedContactRecovery(cfg)
    planner = (ContactRecoveryPlanner(h, graph) if recovery_mode == 'wrist' else
               FingerReconfigurationPlanner(h, graph) if recovery_mode == 'finger_reconfiguration' else None)
    explicit = None
    if combined:
        assert finger == 0 and source['provenance']['source_trace']['relative_time_s'] == 240.
        pivot_config = ExtendedContactRecoveryConfig(angles_deg=(12.,))
        planner = ContactRecoveryPlanner(h, graph, pivot_config)
        explicit = ExplicitSurfaceFingerPlanner(h, graph, target_point_local_m=graph.mesh.triangles_center[9068],
                                                target_face_id=9068)
    initial_q = joint_state(h, data)
    initial_ctrl = np.r_[data.ctrl[h.arm_actuator_ids], data.ctrl[h.hand_actuator_ids]].copy()
    initial_time = float(data.time)
    initial_state = {key: value.copy() for key, value in
                     dict(qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl, qacc_warmstart=data.qacc_warmstart).items()}
    initial_state['time'] = initial_time
    forces, points, normals, active, _, _ = contact_measurement(h, data, .2)
    path, plan_diag = (planner.plan(data, finger, active, forces, normals) if planner is not None else
                       (None, dict(reason='SOLVER_COMPARISON_FIXED_WRIST')))
    if combined:
        assert path is not None and plan_diag['selected']['axis_index'] == 1
        assert abs(plan_diag['selected']['angle_rad']+np.deg2rad(12.)) < 1e-12
    for key in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart'):
        np.testing.assert_array_equal(getattr(data, key), initial_state[key])
    assert data.time == initial_time
    anchor = int(path.anchor_finger) if path is not None else int(np.argmax(np.where(active, forces, -1.)))
    assert active[anchor] and not active[finger]
    np.savez_compressed(directory/'initial.npz', **initial_state)
    np.savez_compressed(directory/'source_frame.npz', **source['frame'])
    (directory/'model.xml').write_bytes(source['model_bytes'])
    (directory/'surface_graph.npz').write_bytes(source['graph_bytes'])
    (directory/'source_run_manifest.json').write_bytes(source['source_manifest_bytes'])
    (directory/'source_clock_anchor.npz').write_bytes(source['clock_anchor_bytes'])
    (directory/'planning.json').write_text(json.dumps(plan_diag, indent=2), encoding='utf-8')
    if path is not None:
        np.savez_compressed(directory/'posture_path.npz', q=path.waypoints,
            segment_durations=path.segment_durations, root_pose=path.root_pose, goal_pose=path.goal_pose,
            anchor_finger=path.anchor_finger, selected_finger=path.selected_finger)
    for name, content in sources.items():
        destination = directory/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    columns = search.columns[finger]+7
    local_columns = search.columns[finger]
    nominal = initial_q.copy()
    unselected_hand = np.setdiff1d(np.arange(7, 23), columns)
    nominal[unselected_hand] = initial_ctrl[unselected_hand]
    if persistent_search:
        mcc = ContactPreservingMCC(h, DT, nominal, initial_ctrl, active, forces)
    raw_path_root = path.waypoints[0] if path is not None else initial_q
    np.testing.assert_array_equal(raw_path_root, initial_q)
    np.savez_compressed(directory/'controller_handoff.npz', raw_path_root_q=raw_path_root,
        prior_control_rad=initial_ctrl, support_preload_delta_rad=initial_ctrl-initial_q,
        applied_initial_reference_rad=nominal, applied_preload_delta_rad=nominal-initial_q,
        selected_columns=columns, unselected_hand_columns=unselected_hand)
    precheck = dict(scope='FK references with retained unselected preload; not physical contact evidence',
                    root_command_reason=guard.command_collision_reason(initial_q, nominal),
                    path_samples=0, failed_samples=[])
    if path is not None:
        for sample_index, sample_time in enumerate(np.linspace(0., path.duration_s, 501)):
            reference = path.sample(sample_time)
            reference[unselected_hand] = initial_ctrl[unselected_hand]
            reason = guard.collision_reason(reference)
            if reason is not None:
                precheck['failed_samples'].append(dict(index=sample_index, time_s=float(sample_time), reason=reason))
        precheck['path_samples'] = 501
    (directory/'preload_geometry_precheck.json').write_text(json.dumps(precheck, indent=2), encoding='utf-8')
    manifest = dict(protocol='i04-anchored-posture-recovery.paired.development.v4' if persistent_search else
                    'i04-anchored-posture-recovery.paired.development.v3' if combined else
                    'i04-anchored-posture-recovery.paired.development.v2', seed=source['seed'],
        selected_finger=finger, anchor_finger=anchor, source_state=source['provenance'],
        object_position_m=h.object_position_m.tolist(), model_sha256=_digest(source['model_bytes']),
        source_hashes={name: _digest(value) for name, value in sources.items()},
        initial_sensor_errors=sensor_errors, initial_state_restore_count=1,
        coverage_authority='NONE', coverage_counted=False, whole_traversal_evidence=False,
        observation_duration_s=duration_s, recovery_mode=recovery_mode,
        intervention_enabled=intervention_enabled, posture_enabled=wrist_enabled,
        search_implementation=search_type.__name__,
        mcc_implementation=type(mcc).__name__, persistent_search=persistent_search,
        common_controls=dict(finger_search_scope='selected finger only',
            mcc_scope='all physically active fingers, including selected after recontact', desired_force_n=2.,
            hand_nominal='selected measured hand; unselected hand retains initial prior control preload; fresh identical MCC states',
            selected_contact_handoff='first actual selected safe contact latches its prior applied command only; then measured MCC',
            off_arm_reference='initial measured arm held fixed',
            maximum_command_speed_rad_s=1.2, stable_confirmation_s=.15,
            search_log_semantics='selected proposal before MCC and command guards; measured q and sensors are actual evidence',
            path_pause_without_anchor_s=.06 if recovery_mode in ('wrist', 'pivot_explicit_target') else None,
            fixed_wrist_finger_path_allowed_without_anchor=recovery_mode in ('finger_reconfiguration', 'pivot_explicit_target'),
            pause_path_when_selected_safe=True,
            tracking_full_speed_rad=.004, tracking_pause_rad=.010,
            force_relief='same measured hand release in both cases; wrist path pauses, no synthetic contacts'),
        arm_self_collision=dict(planning_clearance_m=.001, physical_stop_force_n=.5),
        active_finger_search=asdict(search.config),
        posture_planner=(asdict(planner.config) if is_dataclass(planner.config) else vars(planner.config)) if planner else None,
        controller_handoff_sha256=_digest((directory/'controller_handoff.npz').read_bytes()),
        preload_geometry_precheck_sha256=_digest((directory/'preload_geometry_precheck.json').read_bytes()),
        geometric_path_available=path is not None,
        geometric_path_sha256=_digest((directory/'posture_path.npz').read_bytes()) if path is not None else None)
    if combined:
        manifest['combined_strategy'] = dict(target_mesh_face_id=9068, target_point_local_m=explicit.point_local.tolist(),
            target_maximum_nearest_offset_m=.04, explicit_planner=asdict(explicit.config),
            on='12deg anchored pivot; settle; replan the explicit surface target from actual measured state; selected path then bounded finish',
            off='Fixed measured wrist with bounded local finger search',
            pivot_selected_search_enabled=False, settle_minimum_s=.15, settled_tracking_confirmation_s=.1,
            settled_tracking_threshold_rad=.004, stage2_state_reset=False,
            anchor_loss_policy='pause moving pivot immediately; end moving pivot after 60 ms continuous primary loss; settled stage may use any actual safe anchor',
            force_relief_policy='cancel the combined attempt, preserve measured terminal state')
        if persistent_search:
            manifest['combined_strategy']['local_search']='ActiveFingerSearchV3 persistent physical-loss episode; only 150 ms stable contact resets episode'
            manifest['combined_strategy']['mcc_handoff']='ContactPreservingMCC; search sees held bias; command adds it exactly once'
            manifest['combined_strategy']['requires_rebase']='Cancel stale path; actual-state explicit target replan; at most three granted paths per loss episode'
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    search.rebase(nominal)
    last = initial_ctrl.copy()
    events = [dict(time_s=0., kind='SOURCE_STATE_SENSOR_CHECK', errors=sensor_errors),
              dict(time_s=0., kind='RECOVERY_PROPOSAL', mode=recovery_mode,
                   applied=bool(intervention_enabled and path is not None),
                   reason=plan_diag['reason'], anchor_finger=anchor,
                   duration_s=path.duration_s if path is not None else 0.)]
    logs = {}
    path_elapsed = peak = peak_arm = 0.
    first_touch = stable_since = confirmed = anchor_lost = None
    selected_contact_latched = False
    combo_stage = 'PIVOT' if combined and intervention_enabled else 'OFF_LOCAL'
    target_path = None
    target_elapsed = 0.
    endpoint_since = settled_since = None
    frozen_arm = initial_q[:7].copy()
    reference_replans = []
    if persistent_search and intervention_enabled:
        events.append(dict(time_s=0.,kind='EXPLICIT_PROBE_POSTURE_ATTEMPT',grant=search.begin_posture_attempt(finger)))
    high_steps = 0
    stop = 'RECOVERY_WINDOW_COMPLETE'
    for step in range(int(round(duration_s/DT))+1):
        t = step*DT
        forces, points, normals, active, unsafe, penetration = contact_measurement(h, data, .2)
        arm_contacts = arm_self_contacts(h, data, force_threshold_n=0.)
        arm_force = max([0.]+[c['force_n'] for c in arm_contacts])
        if forces.max() > max(8., peak):
            events.append(dict(time_s=t, kind='FORCE_PEAK_STATE', forces_n=forces.tolist(), **_state(h, data)))
        peak, peak_arm = max(peak, float(forces.max())), max(peak_arm, arm_force)
        high_steps = high_steps+1 if forces.max() > 12. else 0
        hardware = ('SHALLOW_ARM_SELF_COLLISION' if arm_force > .5 else
            'NON_TIP_OR_DEEP_SELF_COLLISION' if unsafe else
            'SUSTAINED_DANGEROUS_FORCE' if high_steps*DT >= .020 else
            'NONFINITE_STATE' if not np.isfinite(data.qpos).all() else
            'JOINT_LIMIT' if np.any(joint_state(h, data) < guard.lower-.02)
                or np.any(joint_state(h, data) > guard.upper+.02) else None)
        safe_selected = bool(active[finger] and forces[finger] <= 8.)
        safe_anchor = bool(active[anchor] and forces[anchor] <= 8.)
        if safe_selected and not selected_contact_latched:
            selected_contact_latched = True
            nominal[columns] = joint_state(h,data)[columns] if persistent_search else last[columns]
            if persistent_search:
                mcc = ContactPreservingMCC(h, DT, nominal, last, active, forces)
            search.rebase(nominal)
            if combined and intervention_enabled:
                combo_stage = 'CONTACT_HOLD'
                frozen_arm = last[:7].copy()
            events.append(dict(time_s=t, kind='SELECTED_CONTACT_REFERENCE_HANDOFF',
                selected_finger=finger, selected_columns=columns.tolist(),
                preserved_prior_control_rad=last[columns].tolist(), measured_q_rad=joint_state(h, data)[columns].tolist()))
        if safe_selected:
            if first_touch is None:
                first_touch = t
                events.append(dict(time_s=t, kind='ACTUAL_SELECTED_RECONTACT', finger=finger))
            stable_since = t if stable_since is None else stable_since
            if confirmed is None and t-stable_since >= .15-1e-12:
                confirmed = t
                events.append(dict(time_s=t, kind='SAME_FINGER_STABLE_RECONTACT', finger=finger,
                                   stable_duration_s=t-stable_since))
        else:
            stable_since = None
        if not safe_anchor and anchor_lost is None:
            anchor_lost = t
            events.append(dict(time_s=t, kind='ACTUAL_ANCHOR_CONTACT_LOST', finger=anchor))
        elif safe_anchor and anchor_lost is not None:
            events.append(dict(time_s=t, kind='ACTUAL_ANCHOR_RECONTACT', finger=anchor,
                               gap_duration_s=t-anchor_lost))
            anchor_lost = None
        in_relief = bool(forces.max() >= 8. or (relief.mode == 'FORCE_RELEASE' and forces.max() >= 4.))
        relief.mode = 'FORCE_RELEASE' if in_relief else 'NORMAL'
        protocol_stop = None
        if combined and intervention_enabled and not hardware:
            if in_relief:
                protocol_stop = 'COMBINED_FORCE_RELEASE_CANCEL'
            elif combo_stage == 'PIVOT' and path_elapsed < path.duration_s and anchor_lost is not None and t-anchor_lost >= .06-1e-12:
                protocol_stop = 'COMBINED_ANCHOR_LOSS_CANCEL'
            if protocol_stop:
                events.append(dict(time_s=t, kind=protocol_stop, forces_n=forces.tolist(), **_state(h, data)))
        if (combined and intervention_enabled and not selected_contact_latched and not hardware
                and not protocol_stop and combo_stage in ('PIVOT', 'PIVOT_SETTLE') and path_elapsed >= path.duration_s):
            combo_stage = 'PIVOT_SETTLE'
            endpoint_since = t if endpoint_since is None else endpoint_since
            endpoint_error = float(np.max(abs(joint_state(h, data)[:7]-path.waypoints[-1, :7])))
            command_tracking_error = float(np.max(abs(joint_state(h, data)[:7]-last[:7])))
            command_endpoint_error = float(np.max(abs(last[:7]-path.waypoints[-1, :7])))
            any_safe_anchor = bool(np.any(active & (forces >= .2) & (forces <= 8.)))
            settle_valid = max(endpoint_error, command_tracking_error, command_endpoint_error) <= .004 and any_safe_anchor and not in_relief
            settled_since = (t if settled_since is None else settled_since) if settle_valid else None
            if (settled_since is not None and t-settled_since >= .1-1e-12
                    and t-endpoint_since >= .15-1e-12 and any_safe_anchor):
                root_state = _state(h, data)
                root_before = {key:getattr(data,key).copy() for key in ('qpos','qvel','ctrl','qacc_warmstart')}
                target_path, target_diag = explicit.plan(data, finger, active, forces, normals)
                for key,value in root_before.items():
                    np.testing.assert_array_equal(getattr(data,key),value)
                assert data.time == root_state['native_time_s']
                nominal = joint_state(h, data)
                frozen_arm = nominal[:7].copy()
                nominal[unselected_hand] = last[unselected_hand]
                search.rebase(nominal)
                mcc = ContactPreservingMCC(h, DT, nominal, last, active, forces) if persistent_search else BasicPassiveMCC(h, DT)
                np.savez_compressed(directory/'stage2_actual_root.npz', **root_before, time=data.time,
                    q_rad=joint_state(h,data), initial_reference_rad=nominal, raw_pivot_endpoint_q=path.waypoints[-1])
                target_precheck = dict(samples=0, failures=[], maximum_command_range_clip_rad=0.)
                if target_path is not None:
                    np.testing.assert_array_equal(target_path.root_q, joint_state(h,data))
                    for n, target_time in enumerate(np.linspace(0., target_path.duration_s, 501)):
                        q_reference = target_path.sample(target_time)
                        q_reference[unselected_hand] = last[unselected_hand]
                        clipped = np.clip(q_reference, guard.lower, guard.upper)
                        target_precheck['maximum_command_range_clip_rad'] = max(target_precheck['maximum_command_range_clip_rad'],
                            float(np.max(abs(q_reference-clipped))))
                        bad = guard.collision_reason(clipped)
                        if bad:
                            target_precheck['failures'].append(dict(index=n,reason=bad))
                    target_precheck['samples'] = 501
                    np.savez_compressed(directory/'explicit_target_path.npz', q=target_path.waypoints,
                        segment_durations=target_path.segment_durations, root_pose=target_path.root_pose,
                        goal_pose=target_path.goal_pose, selected_finger=finger, anchor_finger=target_path.anchor_finger)
                target_diag['preload_reference_precheck'] = target_precheck
                (directory/'stage2_planning.json').write_text(json.dumps(target_diag,indent=2),encoding='utf-8')
                if target_precheck['failures']:
                    target_path = None
                    combo_stage = 'TARGET_PRECHECK_FAILED_LOCAL'
                else:
                    combo_stage = 'EXPLICIT_FINGER_PATH' if target_path is not None else 'TARGET_PLAN_FAILED_LOCAL'
                events.append(dict(time_s=t, kind='ACTUAL_PIVOT_ENDPOINT_REPLAN', accepted=target_path is not None,
                    target_face_id=9068, target_point_local_m=explicit.point_local.tolist(),
                    endpoint_tracking_error_rad=endpoint_error, reason=target_diag['reason'],
                    command_tracking_error_rad=command_tracking_error,
                    command_endpoint_error_rad=command_endpoint_error,
                    endpoint_settle_elapsed_s=t-endpoint_since, tracking_confirmation_s=t-settled_since,
                    target_path_duration_s=target_path.duration_s if target_path is not None else None,
                    raw_pivot_endpoint_q_rad=path.waypoints[-1].tolist(),
                    forces_n=forces.tolist(), native_plant_unchanged=True, **root_state))
        tracking_error = float(np.max(abs(joint_state(h, data)[:7]-last[:7])))
        path_tracking_error = (max(tracking_error, float(np.max(abs(joint_state(h, data)[columns]-last[columns]))))
                               if recovery_mode == 'finger_reconfiguration' or combo_stage == 'EXPLICIT_FINGER_PATH' else tracking_error)
        tracking_scale = float(np.clip((.010-path_tracking_error)/(.010-.004), 0., 1.))
        anchor_ready = safe_anchor if combined else safe_anchor or (anchor_lost is not None and t-anchor_lost < .06)
        finger_path_active = bool(recovery_mode == 'finger_reconfiguration' and intervention_enabled
                                  and path is not None and not selected_contact_latched
                                  and path_elapsed < path.duration_s)
        moving = bool(intervention_enabled and path is not None and path_elapsed < path.duration_s
                      and (recovery_mode == 'wrist' or finger_path_active)
                      and (anchor_ready or recovery_mode == 'finger_reconfiguration')
                      and not in_relief and not safe_selected)
        nominal[:7] = path.sample(path_elapsed)[:7] if wrist_enabled and path is not None else initial_q[:7]
        if recovery_mode == 'finger_reconfiguration' and intervention_enabled and path is not None and not selected_contact_latched:
            nominal[columns] = path.sample(path_elapsed)[columns]
        search_enabled = not in_relief and not finger_path_active
        if combined:
            if not intervention_enabled:
                nominal[:7] = initial_q[:7]
                moving = False
                search_enabled = not in_relief
            elif combo_stage in ('PIVOT','PIVOT_SETTLE'):
                nominal[:7] = path.sample(path_elapsed)[:7]
                moving = bool(combo_stage == 'PIVOT' and anchor_ready and not in_relief and path_elapsed < path.duration_s)
                search_enabled = False
            else:
                nominal[:7] = frozen_arm
                finger_path_active = bool(combo_stage == 'EXPLICIT_FINGER_PATH' and target_path is not None
                                          and target_elapsed < target_path.duration_s and not selected_contact_latched)
                if target_path is not None and not selected_contact_latched:
                    nominal[columns] = target_path.sample(target_elapsed)[columns]
                moving = finger_path_active and not in_relief and not safe_selected
                search_enabled = not in_relief and not finger_path_active
                if combo_stage == 'EXPLICIT_FINGER_PATH' and not finger_path_active and not selected_contact_latched:
                    combo_stage = 'EXPLICIT_TARGET_LOCAL_FINISH'
        handoff_bias = np.r_[np.zeros(7),mcc.held_bias_rad] if persistent_search else np.zeros(23)
        search_nominal = nominal+handoff_bias
        if persistent_search:
            search_mask = np.zeros(4,dtype=bool); search_mask[finger] = True
            proposed, diag = search.update(data, search_nominal, forces, normals, active,
                enabled=search_enabled, search_mask=search_mask)
        else:
            proposed, diag = search.update(data, search_nominal, forces, normals, active, enabled=search_enabled)
        finger_nominal = search_nominal.copy()
        finger_nominal[columns] = proposed[columns]
        if persistent_search and diag['requires_rebase'][finger]:
            # Keep this tick's truthful held proposal, discard the stale path,
            # and build any replacement from the current physical state.
            replan_index = len(reference_replans)+1
            replan_directory = directory/'reference_replans'/f'{replan_index:02d}'
            replan_directory.mkdir(parents=True,exist_ok=False)
            replan_root = _state(h,data)
            native_before = {key:getattr(data,key).copy() for key in ('qpos','qvel','ctrl','qacc_warmstart')}
            nominal = joint_state(h,data); frozen_arm = nominal[:7].copy()
            if not intervention_enabled:
                nominal[:7] = initial_q[:7]
            nominal[unselected_hand] = last[unselected_hand]
            search.rebase(nominal)
            mcc = ContactPreservingMCC(h,DT,nominal,last,active,forces)
            target_path = None; target_elapsed = 0.; moving = False
            selected_contact_latched = False
            grant = dict(accepted=False,reason='FIXED_WRIST_CONTROL_REBASE_ONLY')
            replan_diag = dict(reason='FIXED_WRIST_CONTROL_REBASE_ONLY')
            if intervention_enabled:
                target_path,replan_diag = explicit.plan(data,finger,active,forces,normals)
                if target_path is not None:
                    failures=[]
                    for sample_time in np.linspace(0.,target_path.duration_s,501):
                        reference=target_path.sample(sample_time); reference[unselected_hand]=last[unselected_hand]
                        bad=guard.collision_reason(np.clip(reference,guard.lower,guard.upper))
                        if bad: failures.append(bad)
                    replan_diag['preload_precheck_failures']=failures
                    if failures: target_path=None
                if target_path is not None:
                    grant = search.begin_posture_attempt(finger)
                    if not grant['accepted']:
                        target_path = None
            for key,value in native_before.items():
                np.testing.assert_array_equal(getattr(data,key),value)
            assert data.time == replan_root['native_time_s']
            np.savez_compressed(replan_directory/'actual_root.npz',**native_before,time=data.time)
            if target_path is not None:
                np.testing.assert_array_equal(target_path.root_q,joint_state(h,data))
                np.savez_compressed(replan_directory/'path.npz',q=target_path.waypoints,
                    segment_durations=target_path.segment_durations)
            (replan_directory/'planning.json').write_text(json.dumps(replan_diag,indent=2),encoding='utf-8')
            combo_stage = 'EXPLICIT_FINGER_PATH' if target_path is not None else 'REFERENCE_REBASED_LOCAL'
            event=dict(time_s=t,kind='REFERENCE_REBASE_ACTUAL_REPLAN',index=replan_index,
                accepted=target_path is not None,grant=grant,solver=diag['solver'][finger],
                target_face_id=9068,**replan_root)
            events.append(event); reference_replans.append(event)
        command = mcc.command(data, finger_nominal-(np.r_[np.zeros(7),mcc.held_bias_rad] if persistent_search else 0.), forces, normals, active, 2.)
        if in_relief:
            command = relief.relieve_force(h, data, command, forces, normals)
        before_limits = command.copy()
        command = np.clip(command, last-.0024, last+.0024)
        command = np.clip(command, guard.lower, guard.upper)
        reason = guard.command_collision_reason(joint_state(h, data), command) if not hardware else None
        if reason is not None:
            stop = 'COMMAND_GEOMETRY_STOP'
            events.append(dict(time_s=t, kind=stop, reason=reason, proposed_command_rad=command.tolist(), **_state(h, data)))
        if hardware:
            stop = hardware
            events.append(dict(time_s=t, kind='HARDWARE_STOP', reason=hardware, forces_n=forces.tolist(),
                               arm_self_contacts=arm_contacts, other_unsafe_contacts=unsafe, **_state(h, data)))
        if protocol_stop:
            stop = protocol_stop
        terminal = hardware is not None or reason is not None or protocol_stop is not None or step == int(round(duration_s/DT))
        if terminal:
            command = last.copy()
        applied_search = np.zeros(4, dtype=bool)
        applied_search[finger] = bool(diag['seeking'][finger] and not terminal)
        applied_bias, applied_delta = np.zeros(16), np.zeros(16)
        applied_bias[local_columns] = np.asarray(diag['joint_bias_rad'])[local_columns]
        if not terminal:
            applied_delta[local_columns] = np.asarray(diag['search_delta_rad'])[local_columns]
        states = np.full(4, 'DISABLED', dtype='<U32')
        states[finger] = diag['state'][finger]
        phase = ('FORCE_RELIEF' if in_relief else 'SELECTED_CONTACT_HOLD' if safe_selected else
            'ANCHOR_PAUSE' if not anchor_ready and recovery_mode == 'wrist' else 'TRACKING_PAUSE' if tracking_scale == 0 else
            'FINGER_RECONFIGURATION' if moving and finger_path_active else 'POSTURE_PIVOT' if moving else
            'RECOVERY_PATH_HOLD' if intervention_enabled and path is not None else 'FIXED_WRIST')
        if combined:
            phase = 'FORCE_RELIEF' if in_relief else combo_stage
        values = dict(time_s=t, native_time_s=float(data.time), q_rad=joint_state(h, data),
            qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs], data.qvel[h.hand_dof_adrs]],
            sensor_command_rad=last.copy(), command_rad=command.copy(), qacc_warmstart=data.qacc_warmstart.copy(),
            nominal_reference_rad=nominal.copy(), selected_search_reference_rad=finger_nominal.copy(),
            finger_search_nominal_rad=search_nominal.copy(), mcc_held_bias_rad=(mcc.held_bias_rad if persistent_search else np.zeros(16)),
            finger_search_requires_rebase=np.asarray(diag.get('requires_rebase',[False]*4)),
            command_before_limits_rad=before_limits,
            palm_pose_world=pose(h, data), tip_positions_world_m=tip_positions(h, data),
            contact_positions_world_m=points.copy(), normals_world=normals.copy(), forces_n=forces.copy(),
            active=active.copy(), arm_self_force_n=arm_force, arm_tracking_error_rad=tracking_error,
            path_tracking_error_rad=path_tracking_error,
            maximum_penetration_m=penetration, phase=phase, command_phase=phase,
            posture_path_elapsed_s=path_elapsed, posture_path_advance_scale=tracking_scale if moving else 0.,
            anchor_finger=anchor, selected_finger=finger, coverage=0., covered_goals=0, goal_id=-1,
            finger_search_active=applied_search, finger_search_enabled=search_enabled,
            selected_path_active=finger_path_active, selected_contact_reference_latched=selected_contact_latched,
            combined_stage=combo_stage, explicit_target_path_elapsed_s=target_elapsed,
            finger_search_bias_rad=applied_bias, finger_search_delta_rad=applied_delta,
            finger_search_state=states, finger_search_loss_age_s=np.asarray(diag['loss_age_s']),
            finger_search_distance_m=np.asarray(diag['search_distance_m']),
            finger_search_target_positions_world_m=np.asarray(diag['target_positions_world_m']),
            finger_search_target_normals_world=np.asarray(diag['target_normals_world']),
            finger_search_target_valid=np.asarray(diag['target_valid']),
            finger_search_measured_active=active.copy(),
            proposed_all_finger_search_active=np.asarray(diag['seeking']),
            proposed_all_finger_search_bias_rad=np.asarray(diag['joint_bias_rad']))
        if persistent_search:
            for source_key,log_key in (('search_elapsed_s','finger_search_elapsed_s'),
                ('contact_stable_age_s','finger_contact_stable_age_s'),('posture_attempt_id','finger_posture_attempt_id'),
                ('posture_attempts_this_episode','finger_posture_attempts_this_episode')):
                values[log_key]=np.asarray(diag[source_key])
        for key, value in values.items():
            logs.setdefault(key, []).append(value)
        if terminal:
            break
        if moving:
            if combined and combo_stage == 'EXPLICIT_FINGER_PATH':
                target_elapsed = min(target_path.duration_s, target_elapsed+DT*tracking_scale)
            else:
                path_elapsed = min(path.duration_s, path_elapsed+DT*tracking_scale)
        last = command.copy()
        set_control(h, data, command)
        mujoco.mj_step(h.model, data)
        mujoco.mj_forward(h.model, data)
    arrays = {key: np.asarray(value) for key, value in logs.items()}
    np.savez_compressed(directory/'trace.npz', **arrays)
    np.savez_compressed(directory/'final_state.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl,
                        qacc_warmstart=data.qacc_warmstart, time=data.time)
    (directory/'events.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
    observed = confirmed is not None
    summary = dict(protocol=manifest['protocol'], selected_finger=finger, posture_enabled=wrist_enabled,
        recovery_mode=recovery_mode, intervention_enabled=intervention_enabled,
        search_implementation=search_type.__name__,
        stop_reason=stop, executed_s=t, native_initial_time_s=initial_time, native_final_time_s=float(data.time),
        source_relative_time_s=source['provenance']['source_trace']['relative_time_s'],
        success=observed and stop == 'RECOVERY_WINDOW_COMPLETE', stable_recontact_observed=observed,
        first_recontact_delay_s=first_touch, stable_confirmation_delay_s=confirmed,
        observed_outcome=('INCOMPLETE_GUARD_STOP' if stop != 'RECOVERY_WINDOW_COMPLETE' else
                          'STABLE_RECONTACT_OBSERVED' if observed else 'NO_STABLE_RECONTACT_WITHIN_OBSERVATION'),
        final_selected_safe_contact=safe_selected, final_selected_force_n=float(forces[finger]),
        peak_force_n=peak, peak_arm_self_force_n=peak_arm,
        geometric_path_available=path is not None, planned_path_duration_s=path.duration_s if path else None,
        explicit_target_path_available=target_path is not None, explicit_target_path_elapsed_s=target_elapsed,
        final_combined_stage=combo_stage,
        reference_replans=len(reference_replans),
        executed_path_time_s=path_elapsed, initial_sensor_errors=sensor_errors,
        maximum_arm_command_change_rad=float(np.max(abs(arrays['command_rad'][:, :7]-initial_q[:7]))),
        maximum_actual_arm_speed_rad_s=float(np.max(abs(arrays['qvel_rad_s'][:, :7]))),
        source_state_restores_after_initialization=0, coverage_counted=False, whole_traversal_evidence=False,
        native_recorded_states=len(arrays['time_s']), wall_s=perf_counter()-started,
        dynamic_artifact_hashes={name: _digest((directory/name).read_bytes()) for name in
            ('stage2_actual_root.npz', 'stage2_planning.json', 'explicit_target_path.npz') if (directory/name).exists()},
        reference_replan_artifact_hashes={p.relative_to(directory).as_posix():_digest(p.read_bytes())
            for p in sorted((directory/'reference_replans').rglob('*')) if p.is_file()},
        trace_sha256=_digest((directory/'trace.npz').read_bytes()))
    (directory/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('POSTURE_RECOVERY_PROBE', json.dumps(summary), flush=True)
    return summary


def run_pair(directory, source_directory, *, source_time_s=240., selected_finger=0, duration_s=25.,
             recovery_mode='wrist', persistent_search=False):
    source = prepare_source(source_directory, source_time_s, selected_finger)
    sources = snapshot_sources()
    directory = Path(directory)
    results = {case: run(directory/case, source, posture_enabled=enabled, duration_s=duration_s, sources=sources,
                        recovery_mode=recovery_mode, persistent_search=persistent_search)
               for case, enabled in (('ON', True), ('OFF', False))}
    with np.load(directory/'ON/initial.npz') as on, np.load(directory/'OFF/initial.npz') as off:
        for key in on.files:
            np.testing.assert_array_equal(on[key], off[key])
    result = dict(scope='Controlled source-state posture recovery only; no traversal coverage',
                  identical_initial_state=True, source_state=source['provenance'], cases=results)
    (directory/'paired_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-time', type=float, default=240.)
    parser.add_argument('--finger', type=int, choices=range(4), default=0)
    parser.add_argument('--duration', type=float, default=25.)
    parser.add_argument('--mode', choices=('wrist', 'bounded_only', 'finger_reconfiguration', 'pivot_explicit_target'), default='wrist')
    parser.add_argument('--persistent-search',action='store_true')
    args = parser.parse_args()
    run_pair(args.output, args.source, source_time_s=args.source_time,
             selected_finger=args.finger, duration_s=args.duration, recovery_mode=args.mode,
             persistent_search=args.persistent_search)


if __name__ == '__main__':
    main()
