"""Continuous mustard scan with measured per-finger coordinated recovery.

Geometry produces Cartesian poses. Only native MuJoCo contacts earn coverage.
The old whole-surface experiment and its frozen controller are not modified.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, asdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from time import perf_counter

import mujoco
import numpy as np
from scipy.sparse.csgraph import dijkstra
import trimesh

from .contracts import ExperimentConfig, WristTarget, WholeHandTarget
from .physics import (pose, rotation, quaternion, joint_state, set_control,
                      tip_positions, contact_measurement, BasicPassiveMCC, set_joint_state)
from .full_surface import (build_scene, acquire, FullSurfacePlanner,
    FullSurfaceOracle, pad_support_vector, endpoint_arm_offset, shifted_target, complete_surface)
from .oracle import MeasuredCoverage, align_vector
from .recovery import SharedContactRecovery
from .sliding_metrics import analyze_trace
from .active_fingers_v3 import ActiveFingerSearchV3 as ActiveFingerSearch
from .reference_force_release import applied_reference_force_release
from .planning import SharedGeometricTrajectoryPlanner
from .sliding_safety import SlidingSafetyPlanner, arm_self_contacts


from .surface_slide import (SurfaceStrokePlanner, GraceMCC, LocalProgressBudget, unit,
    reanchor_transition, tracking_progress_scale, arm_force_release)
from .finger_participation import FingerParticipationMonitor, FingerParticipationConfig
from .finger_reconfiguration_planner import FingerReconfigurationPlanner
from .contact_recovery_planner import ContactRecoveryPlanner
from .finger_recovery_execution import FingerRecoveryExecution
from .contact_handoff import ContactPreservingMCC
from .multi_target_finger_planner import MultiTargetFingerPlanner
from .extended_contact_recovery import ExtendedContactRecoveryConfig
from .pivot_handoff_gate import PivotHandoffGate
from .recovery_context import RecoveryContextObserver
from .recovery_queue import RecoveryRequestQueue
from .prepared_surface_transition import plan_surface_transition_with_prepared
from .endpoint_recovery_policy import EndpointRecoveryPolicy


def loaded_project_module_sources(source_root, modules=None):
    """Identify already loaded project Python files; do not predict lazy imports."""
    source_root = Path(source_root).resolve()
    loaded = {}
    for name, module in tuple((sys.modules if modules is None else modules).items()):
        filename = getattr(module, '__file__', None)
        if not name.startswith('Module.') or filename is None:
            continue
        path = Path(filename).resolve()
        if path.suffix != '.py' or not path.is_relative_to(source_root):
            continue
        loaded[name] = path.relative_to(source_root).as_posix()
    return dict(sorted(loaded.items()))


def checked_preload_path(planner, path, previous_command, active, forces):
    """Extra command-space check; the physical path is already checked separately."""
    bias = np.zeros(23)
    for finger in np.flatnonzero(active & (forces >= .2) & (forces < 8.)):
        columns = np.arange(7+4*finger, 11+4*finger)
        bias[columns] = previous_command[columns]-path.root_q[columns]
    previous = np.clip(path.root_q + bias, planner.lower, planner.upper)
    count = 0
    maximum_clip = 0.
    for start, end in zip(path.waypoints[:-1], path.waypoints[1:]):
        samples = max(3, int(np.ceil(np.max(abs(end-start))/.004))+1)
        for fraction in np.linspace(0., 1., samples):
            raw_command = start+fraction*(end-start)+bias
            # Match the existing final actuator reference clamp. Measured
            # unselected joints can sit microradians outside a planning margin
            # while still comfortably inside their physical joint ranges.
            command = np.clip(raw_command, planner.lower, planner.upper)
            maximum_clip = max(maximum_clip, float(np.max(abs(command-raw_command))))
            count += 1
            if not planner.edge_free(previous, command):
                return False, dict(checked_states=count, reason='PRELOADED_COMMAND_PATH_REJECTED',
                    maximum_existing_joint_clip_rad=maximum_clip,
                    held_hand_reference_bias_rad=bias[7:].tolist(),
                    interpretation='Servo reference geometry, separate from measured contact')
            previous = command
    return True, dict(checked_states=count, reason='PRELOADED_COMMAND_PATH_PASSED',
        maximum_existing_joint_clip_rad=maximum_clip,
        held_hand_reference_bias_rad=bias[7:].tolist(),
        interpretation='Servo reference geometry, separate from measured contact')


def run(directory, *, cell='B-MCC', seed=7, budget_s=120., speed=.006, expected_runtime_sha256=None,
        witness_path=None):
    if not np.isfinite(budget_s) or budget_s <= 0 or not 0 < speed <= .012:
        raise ValueError('Positive duration and surface speed in (0, 0.012] m/s required')
    if cell != 'B-MCC':
        raise ValueError('Active surface-target finger planning is B-MCC only; old A-MCC remains frozen')
    source_root = Path(__file__).resolve().parents[2]
    sources = ['Module/i04_geometric_traversal/'+name for name in
        ('release_surface_slide.py', 'surface_slide.py', 'full_surface.py', 'physics.py', 'contracts.py', 'oracle.py',
         'planning.py', 'recovery.py', 'fixtures.py', 'sliding_metrics.py', 'active_fingers.py', 'sliding_safety.py',
         'active_fingers_v3.py', 'reference_force_release.py', 'bounded_finger_step.py', 'finger_participation.py',
         'finger_reconfiguration_planner.py', 'contact_recovery_planner.py', 'finger_recovery_execution.py', 'contact_handoff.py',
         'explicit_surface_finger_planner.py', 'multi_target_finger_planner.py', 'extended_contact_recovery.py',
         'pivot_handoff_gate.py', 'recovery_context.py', 'recovery_queue.py',
         'prepared_hand_transition.py', 'prepared_surface_transition.py', 'endpoint_recovery_policy.py')]
    sources += ['Module/fr3_leap/model.py', 'Module/module_2_fingertip_mcc/controller.py']
    sources += ['Module/i04_geometric_traversal/coordinated_surface_slide.py']
    explicit_source_count = len(set(sources))
    loaded_module_sources = loaded_project_module_sources(source_root)
    sources = sorted(set(sources).union(loaded_module_sources.values()))
    source_bytes = {name: (source_root/name).read_bytes() for name in sources}
    hashes = {name: sha256(content).hexdigest() for name, content in source_bytes.items()}
    if (expected_runtime_sha256 is not None and
            hashes['Module/i04_geometric_traversal/release_surface_slide.py'] != expected_runtime_sha256):
        raise RuntimeError('Controller changed before experiment startup')
    directory = Path(directory)
    if (directory/'manifest.json').exists():
        raise FileExistsError('Use a fresh output directory')
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    h, graph, q, quality = build_scene(directory)
    data, initial = acquire(h, graph, q, seed)
    config = ExperimentConfig(seed=seed, duration_s=budget_s+5., wrist_speed_m_s=.025,
                              wrist_rotation_speed_rad_s=.35, trajectory_joint_speed_rad_s=.8)
    sp = SurfaceStrokePlanner(h, graph, config, cell)
    planner = sp.planner
    oracle = FullSurfaceOracle(h, graph, data)
    oracle.planner = SlidingSafetyPlanner(h, config)
    coverage = MeasuredCoverage(graph, config)
    mcc = ContactPreservingMCC(h, .002)
    relief = SharedContactRecovery(config)
    fingers = ActiveFingerSearch(h, graph, dt=.002)
    participation = FingerParticipationMonitor(h.hand_joint_ranges_rad,
        # Observe persistent finger loss while sliding, but defer any complete
        # posture path until a measured natural endpoint or local fallback.
        config=FingerParticipationConfig(surface_phases=('SLIDE', 'LOCAL_SEARCH')))
    recovery_context = RecoveryContextObserver()
    recovery_queue = RecoveryRequestQueue()
    endpoint_recovery = EndpointRecoveryPolicy()
    context_native_window = deque(maxlen=76)
    finger_reconfigure = MultiTargetFingerPlanner(h, graph)
    wrist_reconfigure = ContactRecoveryPlanner(h, graph, ExtendedContactRecoveryConfig())
    coordinated = None
    coordinated_request = None
    coordinated_result = None
    coordinated_gate = None
    coordinated_gate_diag = None
    coordinated_stage = 0
    coordinated_attempt_started = None
    participation_diag = None
    recovery_sequence = 0
    if witness_path is None:
        witness_path = source_root/'Module/generated/mustard_true_ellipsoid_geometry07/pose_witnesses.npz'
    else:
        witness_path = Path(witness_path).resolve()
        if not witness_path.is_file():
            raise FileNotFoundError(f'Explicit witness cache does not exist: {witness_path}')
    witness_bytes = witness_path.read_bytes() if witness_path.exists() else None
    witnesses = None
    if witness_bytes is not None:
        with np.load(BytesIO(witness_bytes)) as saved_witnesses:
            witnesses = {key: saved_witnesses[key].copy() for key in saved_witnesses.files}
        (directory/'geometric_pose_witnesses.npz').write_bytes(witness_bytes)
    np.savez_compressed(directory/'initial.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl)
    (directory/'model.xml').write_text(h.xml, encoding='utf-8')
    np.savez_compressed(directory/'surface_graph.npz', vertices=graph.vertices_m, faces=graph.faces,
                        normals=graph.normals, required_vertices=graph.required_vertices)
    for name, content in source_bytes.items():
        dest = directory/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(content)
    manifest = dict(protocol='i04-mustard-coordinated-finger-sliding.development.v3.8-release', cell=cell, seed=seed,
        object_position_m=h.object_position_m.tolist(), budget_s=budget_s,
        speed_m_s=speed, source_hashes=hashes, initial_state=initial, geometry_quality=quality,
        dp='DEFERRED_BY_USER', full_surface_claim=False,
        runtime_entrypoint='Module/i04_geometric_traversal/release_surface_slide.py',
        active_finger_source='Module/i04_geometric_traversal/active_fingers_v3.py',
        force_release_source='Module/i04_geometric_traversal/reference_force_release.py',
        source_lineage=dict(parent_entry='Module/i04_geometric_traversal/coordinated_surface_slide.py',
            parent_entry_sha256='e1f2b0bfc875f24637e67b2e354e21bebb6c92a67b783cc1168e87b45a4407e3'),
        source_snapshot_policy=dict(scope='Explicit runtime files plus loaded project module snapshot',
            explicit_source_count=explicit_source_count, loaded_module_sources=loaded_module_sources,
            capture_time='Run startup before build_scene/acquire', future_dynamic_imports_included=False),
        applied_reference_relief=dict(reference='Actual prior guarded command',
            original_site_jacobian_and_damping=True, original_slew_range_force_guards=True,
            arm_handoff_unchanged=True, loss_after_withdrawal_requires_measured_recovery=True),
        causal_scope='Single control change relative to v3.6: finger force-release reference origin',
        coordinated_recovery=dict(observer=asdict(participation.config),
            measured_geometry_context=asdict(recovery_context.config),
            deferred_request_queue=asdict(recovery_queue.config),
            anchored_endpoint_escalation=asdict(endpoint_recovery.config),
            finger_planner=asdict(finger_reconfigure.finger_config),
            surface_target_selection=asdict(finger_reconfigure.config), wrist_planner=asdict(wrist_reconfigure.config),
            priority=['FINISH_ADVANCING_SURFACE_STROKE', 'SHORT_FIXED_WRIST_AT_NATURAL_ENDPOINT',
                      'FINITE_ANCHORED_ENDPOINT_AFTER_REPEATED_FIXED_FAILURE',
                      'SAFE_SURFACE_CONTINUATION', 'LOCAL_SEARCH_FULL_RECOVERY_FALLBACK'],
            coordinated_recovery_allowed_phase='NATURAL_ENDPOINT_SHORT_FIXED_OR_MEASURED_ANCHORED_ESCALATION_OR_LOCAL_SEARCH_FALLBACK',
            contact_confirmation_s=.15, planned_path_is_not_contact=True,
            pivot_followup='One current-measurement finger replan after actual pivot endpoint and settling',
            pivot_settle_s=.15, pivot_tracking_stable_s=.10, pivot_tracking_limit_rad=.004,
            maximum_total_attempt_s=60., recovery_counts_toward_local_no_gain_budget=True),
        active_fingers=True, completion_policy='All original vertices and required goals measured, then 160ms safe hold',
        active_finger_search=asdict(fingers.config),
        active_finger_lifecycle=dict(loss_history_survives_rebase=True, air_pauses_local_budget=True,
            stable_measured_contact_reset_s=fingers.stable_contact_s,
            maximum_posture_attempts_per_measured_context=fingers.maximum_posture_attempts_per_loss_episode,
            attempt_budget_reset='Accepted coordinated path or explicit novel measured context; never stroke rebase',
            context_renews_only_exhausted_budget=True, true_loss_history_preserved=True),
        mcc_handoff=dict(retain='Last applied minus new nominal, actual safe hand contacts only',
            arm_bias=False, intentional_air_clears=True, original_mcc_limits_unchanged=True,
            coordinated_path_preload_geometry_checked=True),
        witness_path=str(witness_path.resolve()) if witnesses is not None else None,
        witness_snapshot='geometric_pose_witnesses.npz' if witness_bytes is not None else None,
        witness_sha256=sha256(witness_bytes).hexdigest() if witness_bytes is not None else None,
        plan=dict(surface_spacing_m=.002, minimum_stroke_m=.015, fallback_stroke_m=.005, preferred_stroke_m=.11,
                  local_no_gain_stroke_limit=6, local_no_gain_time_limit_s=45.,
                  contact_loss_grace_s=.1, tangent_pause_loss_s=.06, recovery_timeout_s=2.,
                  recovery_bound_m=.006, geometry_lift_clearance_m=[.004,.008,.012,.025,.04],
                  transfer_speed_m_s=.025, transfer_rotation_rad_s=.35),
        tracking=dict(full_speed_error_rad=.004, pause_error_rad=.010,
                      replan_after_pause_s=.30),
        prepared_surface_fallback=dict(original_surface_transition_first=True,
            invoke_only_after_original_failure=True, maximum_goals=8, pose_variants=[0,1,2,3],
            retreat_distances_m=[.004,.008,.012,.025,.04], short_route_s=25.,
            selection='Shortest safe route among finite tested candidates',
            preparation_controller_mode='Intentional air for the complete validated composite path',
            actual_contact_required_for_coverage=True),
        arm_self_collision=dict(planning_clearance_m=.001, physical_stop_force_n=.5,
                                excludes='same body, directly adjacent or same welded body'),
        coverage=dict(radius_m=.025, residual_limit_m=.005, normal_cosine=.5, force_threshold_n=.2))
    if witness_bytes is None:
        manifest.pop('witness_snapshot')
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    logs, events = {}, []
    last = np.r_[data.ctrl[h.arm_actuator_ids], data.ctrl[h.hand_actuator_ids]]
    nominal = joint_state(h, data)
    phase, stroke, stroke_id, s, offset = 'SELECT', None, -1, 0., 0.
    normal = oracle.initial_normal.copy()
    normal_in_wrist = rotation(pose(h, data)[3:]).inv().apply(normal)
    loss_start = None
    stable_since = None
    failed_starts = 0
    transfer = None
    transfer_elapsed = 0.
    transfer_queue = []
    lift_attempts = np.zeros(len(graph.vertices_m))
    high_run = 0
    peak = 0.
    peak_arm_self_force = 0.
    stop = 'TIME_BUDGET_REACHED'
    phase_started = 0.
    local_progress = LocalProgressBudget()
    pending_stroke = None
    full_since = None
    completion_nominal = None
    reacquire_nominal = None
    tracking_pause_since = None
    checkpoint_index = chunk_number = 0
    def checkpoint(t):
        nonlocal checkpoint_index, chunk_number
        chunks = directory/'trace_chunks'; chunks.mkdir(exist_ok=True)
        arrays = {key: np.asarray(value[checkpoint_index:]) for key,value in logs.items()}
        if len(arrays.get('time_s', [])):
            np.savez_compressed(chunks/f'{chunk_number:04d}.npz', **arrays)
            chunk_number += 1; checkpoint_index = len(logs['time_s'])
        np.savez_compressed(directory/'coverage_checkpoint.tmp.npz',
            covered_vertices=coverage.visited_vertices, required_goal_visited=coverage.visited_goals,
            qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl, qacc_warmstart=data.qacc_warmstart,
            time=data.time, elapsed=t)
        (directory/'coverage_checkpoint.tmp.npz').replace(directory/'coverage_checkpoint.npz')
    for step in range(int(budget_s/.002)+1):
        t = step*.002
        natural_endpoint_this_tick = False
        observed_phase = phase
        forces, points, normals, active, unsafe, penetration = contact_measurement(h, data, .2)
        self_contacts = arm_self_contacts(h, data, force_threshold_n=0.)
        self_force = max([0.] + [item['force_n'] for item in self_contacts])
        peak_arm_self_force = max(peak_arm_self_force, self_force)
        safe = active & (forces <= 8.)
        if forces.max() > max(8., peak):
            events.append(dict(time_s=t, kind='FORCE_PEAK_STATE', phase=phase,
                forces_n=forces.tolist(), q_rad=joint_state(h, data).tolist(),
                qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs], data.qvel[h.hand_dof_adrs]].tolist(),
                sensor_command_rad=last.tolist(), qacc_warmstart=data.qacc_warmstart.tolist()))
        peak = max(peak, float(forces.max()))
        high_run = high_run+1 if forces.max() > 12. else 0
        hard = ('SHALLOW_ARM_SELF_COLLISION' if self_force > .5 else
                'NON_TIP_OBJECT_COLLISION' if unsafe else 'SUSTAINED_DANGEROUS_FORCE'
                if high_run*.002 >= .02 else 'NONFINITE_STATE' if not np.all(np.isfinite(data.qpos))
                else 'JOINT_LIMIT' if np.any(joint_state(h, data) < planner.lower-.02)
                or np.any(joint_state(h, data) > planner.upper+.02) else None)
        if hard:
            stop = hard
            events.append(dict(time_s=t, kind='HARDWARE_STOP', reason=hard, non_tip=unsafe,
                arm_self_contacts=self_contacts, forces_n=forces.tolist(),
                q_rad=joint_state(h, data).tolist(),
                qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs], data.qvel[h.hand_dof_adrs]].tolist(),
                sensor_command_rad=last.tolist(), qacc_warmstart=data.qacc_warmstart.tolist()))
            break
        if step % 5 == 0:
            coverage.update(points-h.object_position_m, normals, active)
        if complete_surface(coverage):
            if completion_nominal is None:
                completion_nominal = joint_state(h, data)
                mcc = ContactPreservingMCC(h, .002, completion_nominal, last, active, forces,
                    intentional_air=False)
                fingers.rebase(completion_nominal)
                events.append(dict(time_s=t, kind='ALL_SURFACE_MEASURED_PENDING_STABILITY'))
            phase = 'COMPLETION_HOLD'
            nominal = completion_nominal.copy()
            if np.any(safe) and forces.max() <= 8.:
                full_since = t if full_since is None else full_since
            else:
                full_since = None
        measured_count = int(coverage.visited_vertices.sum())
        if phase == 'COMPLETION_HOLD':
            coordinated = coordinated_request = None
        # A request is based on previous measured observations, but every
        # candidate is rooted in the current state and actual safe support.
        if coordinated_request is not None:
            request, coordinated_request = coordinated_request, None
            selected = request['finger']
            short_boundary = request.get('dispatch_mode') == 'SHORT_FIXED_WRIST'
            request_current = (t-request['time_s'] < recovery_queue.config.request_ttl_s-1e-12
                and request.get('dispatch_context_id') == int(fingers.recovery_context_id[selected])
                and request.get('dispatch_episode_id') == recovery_context._episode_id[selected])
            eligible_phase = request_current and (phase == 'SELECT' if short_boundary else phase == 'LOCAL_SEARCH')
            request_deferred_for_slide = False
            if request_current and not short_boundary and phase == 'LOCAL_SEARCH' and not active[selected] and forces.max() < 8. and np.any(safe):
                # The contact geometry may have improved during local search.
                # Give a newly available sliding continuation first priority.
                continuation, failures = sp.plan(data, coverage, forces, points, normals, safe)
                if continuation is not None:
                    pending_stroke = (continuation, failures)
                    phase = 'SELECT'
                    request_deferred_for_slide = True
                    eligible_phase = False
                    events.append(dict(time_s=t, kind='COORDINATED_RECOVERY_DEFERRED_FOR_SURFACE_SLIDE',
                        request=request, measured_recovery_success=False))
            if eligible_phase and not active[selected] and forces.max() < 8. and np.any(safe):
                path, finger_plan = finger_reconfigure.plan(data, selected, active, forces, normals)
                fixed_failure_reason = finger_plan.get('reason') if path is None else None
                if path is not None and short_boundary and path.duration_s > recovery_queue.config.maximum_endpoint_path_duration_s:
                    finger_plan['boundary_rejection'] = 'FIXED_WRIST_PATH_EXCEEDS_ENDPOINT_TIME_LIMIT'
                    finger_plan['rejected_path_duration_s'] = path.duration_s
                    path = None
                if path is not None:
                    preload_ok, preload_check = checked_preload_path(planner, path, last, active, forces)
                    finger_plan['preload_command_path_check'] = preload_check
                    if not preload_ok:
                        path = None
                wrist_plan = None
                path_kind = 'FINGER_RECONFIGURE'
                endpoint_grant = None
                if short_boundary and path is None and fixed_failure_reason is not None:
                    endpoint_grant = endpoint_recovery.consider_fixed_failure(t, int(selected),
                        request['request_id'], fixed_failure_reason, measured_active=active,
                        forces_n=forces, palm_pose_world=pose(h, data), phase=phase,
                        episode_id=int(recovery_context._episode_id[selected]),
                        context_id=int(fingers.recovery_context_id[selected]))
                    if endpoint_grant['accepted']:
                        window = list(context_native_window)[-75:]+[dict(time_s=t,
                            native_time_s=float(data.time), q_rad=joint_state(h, data),
                            qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs], data.qvel[h.hand_dof_adrs]],
                            sensor_command_rad=last.copy(), qacc_warmstart=data.qacc_warmstart.copy(),
                            palm_pose_world=pose(h, data), active=active.copy(), forces_n=forces.copy(),
                            phase=phase)]
                        window_name = f'endpoint_recovery_grant_{endpoint_grant["event_id"]:04d}_native.npz'
                        np.savez_compressed(directory/window_name,
                            **{key: np.asarray([row[key] for row in window]) for key in window[0]})
                        endpoint_grant.update(native_support_window_file=window_name,
                            native_support_window_sha256=sha256((directory/window_name).read_bytes()).hexdigest())
                    events.append(endpoint_grant)
                endpoint_escalation = endpoint_grant is not None and endpoint_grant['accepted']
                if path is None and (not short_boundary or endpoint_escalation):
                    path, wrist_plan = wrist_reconfigure.plan(data, selected, active, forces, normals)
                    path_kind = 'WRIST_PIVOT'
                    if (path is not None and endpoint_escalation and
                            path.anchor_finger not in endpoint_grant['stable_support_fingers']):
                        wrist_plan['endpoint_rejection'] = 'PIVOT_ANCHOR_LACKS_150MS_MEASURED_SUPPORT'
                        path = None
                    if path is not None:
                        preload_ok, preload_check = checked_preload_path(planner, path, last, active, forces)
                        wrist_plan['preload_command_path_check'] = preload_check
                        if not preload_ok:
                            path = None
                events.append(dict(time_s=t, kind='COORDINATED_RECOVERY_PLAN', request=request,
                    finger_plan=finger_plan, wrist_plan=wrist_plan, accepted=path is not None,
                    endpoint_escalation_grant_id=(endpoint_grant['event_id'] if endpoint_escalation else None)))
                if path is not None and path.duration_s+2.3 > 60.:
                    events.append(dict(time_s=t, kind='COORDINATED_RECOVERY_PLAN_TOO_LONG',
                        duration_s=path.duration_s, remaining_attempt_s=60.))
                    path = None
                grant = fingers.begin_posture_attempt(selected) if path is not None else None
                if path is not None and grant['accepted']:
                    recovery_sequence += 1
                    ended = local_progress.finish(measured_count, 'COORDINATED_RECOVERY')
                    if ended is not None:
                        events.append(dict(time_s=t, kind='LOCAL_STROKE_RESULT', **ended))
                    coordinated = FingerRecoveryExecution(path, kind=path_kind, started_s=t)
                    coordinated_stage = 0
                    coordinated_attempt_started = t
                    coordinated_gate = (PivotHandoffGate(path.duration_s, selected, started_s=t,
                                        endpoint_arm_q=path.waypoints[-1][:7], anchor_finger=path.anchor_finger)
                                        if path_kind == 'WRIST_PIVOT' else None)
                    coordinated_gate_diag = None
                    nominal = joint_state(h, data)
                    fingers.rebase(nominal)
                    mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
                    phase, phase_started = 'FINGER_RECOVERY', t
                    pending_stroke = None
                    np.savez_compressed(directory/f'finger_recovery_{recovery_sequence:04d}.npz',
                        q=path.waypoints, durations_s=path.segment_durations,
                        selected_finger=selected, anchor_finger=path.anchor_finger,
                        actual_root_q=joint_state(h, data), qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                        ctrl=data.ctrl.copy(), qacc_warmstart=data.qacc_warmstart.copy(), native_time_s=data.time)
                    events.append(dict(time_s=t, kind='COORDINATED_RECOVERY_BEGIN',
                        sequence=recovery_sequence, selected_finger=selected,
                        anchor_finger=path.anchor_finger, path_kind=path_kind, duration_s=path.duration_s,
                        request_id=request['request_id'], local_search_grant=grant,
                        path_snapshot=f'finger_recovery_{recovery_sequence:04d}.npz',
                        path_snapshot_sha256=sha256((directory/f'finger_recovery_{recovery_sequence:04d}.npz').read_bytes()).hexdigest(),
                        dispatch_mode=('ANCHORED_ENDPOINT_FALLBACK' if endpoint_escalation
                            else request.get('dispatch_mode', 'FULL_FALLBACK')),
                        endpoint_escalation_grant_id=(endpoint_grant['event_id'] if endpoint_escalation else None),
                        endpoint_motion_pause=short_boundary, measured_recovery_success=False))
                elif grant is not None:
                    events.append(dict(time_s=t, kind='COORDINATED_RECOVERY_ATTEMPT_DENIED',
                        request=request, grant=grant))
            elif not request_deferred_for_slide:
                events.append(dict(time_s=t, kind='COORDINATED_RECOVERY_REQUEST_STALE', request=request,
                    phase=phase, request_time_context_current=request_current,
                    measured_active=active.tolist(), forces_n=forces.tolist()))
        local_phases = ('SLIDE', 'SELECT', 'LOCAL_SEARCH', 'FINGER_RECOVERY', 'FINGER_STABILIZE')
        local_progress.observe(measured_count, .002, local=phase in local_phases)
        if local_progress.exhausted and phase in local_phases:
            ended = local_progress.finish(measured_count, 'LOCAL_PROGRESS_BUDGET')
            events.append(dict(time_s=t, kind='LOCAL_PROGRESS_BUDGET_EXHAUSTED',
                ended_stroke=ended, empty_strokes=local_progress.empty_strokes,
                local_without_gain_s=local_progress.local_s))
            phase = 'LIFT_PLAN'  # The transition planner still tries contact-near bridges first.
            pending_stroke = None
            nominal = joint_state(h, data)
            fingers.rebase(nominal); mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
        # Run this before SELECT so a found stroke is installed from the same
        # measured state, without a second mutating planner call or mj_step.
        if phase == 'LOCAL_SEARCH' and t-phase_started >= 2.:
            candidate, failures = sp.plan(data, coverage, forces, points, normals, safe)
            if candidate is not None:
                pending_stroke = (candidate, failures)
                phase = 'SELECT'
                events.append(dict(time_s=t, kind='LOCAL_SEARCH_SUPPORT_FOUND'))
            else:
                phase = 'LIFT_PLAN'
                events.append(dict(time_s=t, kind='LIFT_REQUIRED', reasons=failures))
        if phase == 'SELECT':
            if pending_stroke is not None:
                stroke, failures = pending_stroke
                pending_stroke = None
            else:
                stroke, failures = sp.plan(data, coverage, forces, points, normals, safe)
            if stroke is not None:
                stroke_id += 1; s = 0.; offset = 0.; loss_start = None
                tracking_pause_since = None
                phase = 'SLIDE'; phase_started = t
                local_progress.begin(stroke_id, measured_count)
                nominal = stroke.q[0].copy()
                fingers.rebase(nominal)
                mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
                events.append(dict(time_s=t, kind='SURFACE_STROKE', stroke_id=stroke_id,
                    length_m=stroke.length, points=len(stroke.q), start=stroke.positions[0].tolist(),
                    end=stroke.positions[-1].tolist(), finger=int(stroke.finger), failures=failures,
                    exposed_fields=['wrist_pose'] if cell == 'A-MCC' else ['wrist_pose', 'tip_positions']))
                np.savez_compressed(directory/f'stroke_{stroke_id:04d}.npz', q=stroke.q,
                    wrist=stroke.wrist, surface_positions_local=stroke.positions,
                    normals=stroke.normals, distance=stroke.distances)
            else:
                phase = 'LOCAL_SEARCH'; phase_started = t
                nominal = joint_state(h, data)
                fingers.rebase(nominal)
                mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
                events.append(dict(time_s=t, kind='LOCAL_SEARCH_BEGIN', reasons=failures))
        if phase == 'LIFT_PLAN':
            outward = unit(normals[safe].mean(axis=0)) if np.any(safe) else normal
            result = plan_surface_transition_with_prepared(sp, oracle, data, coverage, lift_attempts, outward, witnesses)
            fallback_diagnostics = sp.last_prepared_transition_diagnostics
            if fallback_diagnostics['fallback_invoked']:
                events.append(dict(time_s=t, kind='PREPARED_SURFACE_FALLBACK_SEARCH',
                    accepted=result is not None, diagnostics=fallback_diagnostics,
                    measured_covered_vertices=measured_count, physical_contact_or_coverage_claim=False))
            if result is None:
                failed_starts += 1
                if failed_starts >= 4:
                    stop = 'NO_SAFE_REPOSITION_AFTER_EXPANDED_SEARCH'; break
                events.append(dict(time_s=t, kind='REPOSITION_SEARCH_EXPANDED', attempt=failed_starts))
                phase = 'RETRY_REPOSITION'; phase_started = t
                nominal = joint_state(h, data)
                mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN')); fingers.rebase(nominal)
            else:
                failed_starts = 0
                local_progress.transition_committed()
                segments, normal, transition_meta = result
                phase, transfer = segments[0]
                transfer_queue = segments[1:]
                transfer_elapsed = 0.; offset = 0.; phase_started = t
                nominal = transfer.root_q.copy()
                mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
                fingers.rebase(joint_state(h, data))
                events.append(dict(time_s=t, kind='NEAR_BRIDGE_COMMIT' if phase == 'NEAR_TRANSFER' else 'LIFT_COMMIT',
                    **transition_meta, exposed_fields=['wrist_pose', 'tip_positions']))
        was_in_relief = relief.mode == 'FORCE_RELEASE'
        in_relief = forces.max() >= 8 or (was_in_relief and forces.max() >= 4)
        relief.mode = 'FORCE_RELEASE' if in_relief else 'NORMAL'
        if in_relief and not was_in_relief:
            events.append(dict(time_s=t, kind='FORCE_RELEASE_BEGIN', phase=phase,
                               forces_n=forces.tolist()))
            mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
        elif was_in_relief and not in_relief:
            events.append(dict(time_s=t, kind='FORCE_RELEASE_REPLAN', phase=phase))
            # The old path caused the overload. Re-root at the measured
            # released state rather than returning to the same compression.
            phase = 'SELECT' if np.any(safe) else 'LIFT_PLAN'
            nominal = joint_state(h, data)
            fingers.rebase(nominal)
            mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
            if complete_surface(coverage):
                completion_nominal = nominal.copy()
            offset = 0.
        if phase == 'SLIDE':
            finger = stroke.finger
            good = bool(safe[finger])
            if good:
                if loss_start is not None:
                    events.append(dict(time_s=t, kind='CONTACT_RECOVERED', duration_s=t-loss_start,
                                       stroke_id=stroke_id, same_arclength_resume_m=s))
                loss_start = None
            elif loss_start is None:
                loss_start = t
                events.append(dict(time_s=t, kind='CONTACT_GAP', stroke_id=stroke_id, arclength_m=s))
            previous_nominal, _ = stroke.sample(s)
            progress_scale, tracking_lag = tracking_progress_scale(joint_state(h, data), last, previous_nominal)
            stalled = progress_scale < .02
            if stalled and tracking_pause_since is None:
                tracking_pause_since = t
                events.append(dict(time_s=t, kind='ARM_TRACKING_PAUSE', error_rad=tracking_lag,
                                   forces_n=forces.tolist(), stroke_id=stroke_id))
            elif not stalled:
                tracking_pause_since = None
            advance = not in_relief and forces.max() < 4. and (good or t-loss_start < .06)
            if advance:
                # Constant arc-speed with a short starting ramp, no stop at
                # every mesh point. Bound the corresponding joint velocity.
                a, _ = stroke.sample(s)
                b, _ = stroke.sample(min(stroke.length, s+speed*.002))
                ratio = min(1., .8*.002/max(float(np.max(abs(b-a))), 1e-12))
                s = min(stroke.length, s+speed*.002*ratio*progress_scale*min(1., (t-phase_started+.002)/.3))
            nominal, normal = stroke.sample(s)
            # Slow common normal trim handles geometry mismatch; fingertip
            # MCC handles faster local force error. No tangential search.
            velocity = float(np.clip(.0005*(2.-forces[finger]), -.002, .0015))
            velocity = min(velocity, float(np.clip(.0005*(3.-forces.max()), -.002, .0015)))
            offset = float(np.clip(offset+velocity*.002, -.004, .006))
            nominal[:7] += endpoint_arm_offset(h, data, normal, offset)
            if tracking_pause_since is not None and t-tracking_pause_since >= .30:
                events.append(dict(time_s=t, kind='ARM_TRACKING_REPLAN', error_rad=tracking_lag,
                                   stroke_id=stroke_id))
                phase = 'SELECT'; nominal = joint_state(h, data)
                fingers.rebase(nominal); mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
                offset = 0.; tracking_pause_since = None
            elif loss_start is not None and t-loss_start >= 2.:
                events.append(dict(time_s=t, kind='LOCAL_RECOVERY_TIMEOUT', stroke_id=stroke_id))
                # Another finger may already support a continuation.
                phase = 'SELECT' if np.any(safe) else 'LOCAL_SEARCH'
                phase_started = t
                nominal = joint_state(h, data)
                fingers.rebase(nominal)
                mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
            elif s >= stroke.length:
                events.append(dict(time_s=t, kind='STROKE_ENDPOINT', stroke_id=stroke_id, safe_contact=good))
                phase = 'SELECT'
                natural_endpoint_this_tick = True
        elif phase in ('FINGER_RECOVERY', 'FINGER_STABILIZE'):
            measured = joint_state(h, data)
            previous_path = coordinated.path.sample(coordinated.elapsed_s)
            tracking_error = max(float(np.max(abs(measured[:7]-last[:7]))),
                                 float(np.max(abs(previous_path[:7]-last[:7]))))
            if coordinated.contact_pose is None:
                selected_columns = np.arange(7+4*coordinated.path.selected_finger,
                                             11+4*coordinated.path.selected_finger)
                tracking_error = max(tracking_error,
                    float(np.max(abs(measured[selected_columns]-last[selected_columns]))))
            coordinated_result = coordinated.update(t, joint_state(h, data), active, forces,
                tracking_error)
            if (coordinated_result['status'] == 'RUNNING'
                    and t-coordinated_attempt_started >= 60.-1e-12):
                coordinated.status, coordinated.reason = 'ABORTED', 'TOTAL_ATTEMPT_TIME_LIMIT'
                coordinated_result.update(status=coordinated.status, reason=coordinated.reason,
                    nominal_q=measured.copy(), rebase_required=True, search_enabled=False,
                    measured_recovery_success=False)
            if coordinated_gate is not None:
                coordinated_gate_diag = coordinated_gate.update(t, coordinated.elapsed_s,
                    measured, last, active, forces, previous_stage_status=coordinated_result['status'])
                if coordinated_gate_diag['needs_finger_replan']:
                    # Stage two is solved only now, from the actual settled
                    # state. It belongs to the same loss episode and grant.
                    selected = coordinated.path.selected_finger
                    remaining = max(0., 60.-(t-coordinated_attempt_started))
                    path, diagnostics = finger_reconfigure.plan(data, selected, active, forces, normals)
                    if path is not None:
                        preload_ok, preload_check = checked_preload_path(planner, path, last, active, forces)
                        diagnostics['preload_command_path_check'] = preload_check
                        diagnostics['remaining_attempt_s'] = remaining
                        diagnostics['required_stage_with_confirmation_s'] = path.duration_s+2.3
                        if not preload_ok or path.duration_s+2.3 > remaining:
                            diagnostics['followup_rejection'] = ('PRELOADED_COMMAND_PATH_REJECTED'
                                if not preload_ok else 'INSUFFICIENT_REMAINING_ATTEMPT_TIME')
                            path = None
                    np.savez_compressed(directory/f'finger_recovery_{recovery_sequence:04d}_stage1_root.npz',
                        actual_root_q=measured, qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                        ctrl=data.ctrl.copy(), qacc_warmstart=data.qacc_warmstart.copy(),
                        native_time_s=data.time, elapsed_s=t, forces_n=forces, active=active,
                        normals_world=normals, points_world=points)
                    events.append(dict(time_s=t, kind='COORDINATED_RECOVERY_ACTUAL_ENDPOINT_REPLAN',
                        sequence=recovery_sequence, selected_finger=selected, stage=1,
                        gate=coordinated_gate_diag, remaining_attempt_s=remaining,
                        diagnostics=diagnostics, accepted=path is not None,
                        new_search_grant=False, measured_recovery_success=False))
                    coordinated_gate = None  # One pivot and at most one follow-up.
                    if path is None:
                        coordinated.status, coordinated.reason = 'ABORTED', 'NO_SAFE_ACTUAL_ENDPOINT_FINGER_PATH'
                        coordinated_result.update(status=coordinated.status, reason=coordinated.reason,
                            nominal_q=measured.copy(), rebase_required=True, search_enabled=False,
                            measured_recovery_success=False)
                    else:
                        coordinated_stage = 1
                        coordinated = FingerRecoveryExecution(path, kind='FINGER_RECONFIGURE', started_s=t)
                        nominal = measured.copy()
                        fingers.rebase(nominal)
                        mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces)
                        coordinated_result = coordinated.update(t, measured, active, forces, tracking_error)
                        np.savez_compressed(directory/f'finger_recovery_{recovery_sequence:04d}_stage1.npz',
                            q=path.waypoints, durations_s=path.segment_durations,
                            selected_finger=selected, anchor_finger=path.anchor_finger)
                        events.append(dict(time_s=t, kind='COORDINATED_RECOVERY_STAGE_BEGIN',
                            sequence=recovery_sequence, stage=1, selected_finger=selected,
                            anchor_finger=path.anchor_finger, path_kind='FINGER_RECONFIGURE',
                            duration_s=path.duration_s, total_attempt_started_s=coordinated_attempt_started,
                            new_search_grant=False, measured_recovery_success=False))
            nominal = coordinated_result['nominal_q'].copy()
            phase = coordinated_result['phase']
            if coordinated_result['rebase_required']:
                fingers.rebase(nominal)
                mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
            if coordinated_result['first_contact']:
                events.append(dict(time_s=t, kind='COORDINATED_RECOVERY_FIRST_CONTACT',
                    sequence=recovery_sequence, selected_finger=coordinated_result['selected_finger'],
                    forces_n=forces.tolist(), path_elapsed_s=coordinated_result['path_elapsed_s']))
            if coordinated_result['status'] != 'RUNNING':
                events.append(dict(time_s=t, kind='COORDINATED_RECOVERY_END', sequence=recovery_sequence,
                    status=coordinated_result['status'], reason=coordinated_result['reason'],
                    selected_finger=coordinated_result['selected_finger'], forces_n=forces.tolist(),
                    stage=coordinated_stage, total_attempt_elapsed_s=t-coordinated_attempt_started,
                    measured_recovery_success=coordinated_result['measured_recovery_success']))
                phase = 'SELECT' if np.any(safe) else 'LOCAL_SEARCH'
                phase_started = t
                coordinated = None
                coordinated_gate = None
        elif phase == 'RETRY_REPOSITION':
            if t-phase_started >= .2:
                phase = 'SELECT' if np.any(safe) else 'LIFT_PLAN'
        elif phase in ('LIFT', 'TRANSFER', 'APPROACH', 'NEAR_TRANSFER'):
            if not in_relief:
                transfer_elapsed += .002
            nominal = transfer.sample(transfer_elapsed)
            if transfer_elapsed >= transfer.duration_s:
                if transfer_queue:
                    phase, transfer = transfer_queue.pop(0)
                    transfer_elapsed = 0.
                    measured = joint_state(h, data)
                    if (np.max(abs(measured-transfer.root_q)) > .02 or
                            not planner.edge_free(measured, transfer.root_q)):
                        revised, failure = reanchor_transition(planner, data, transfer, cell,
                                                              allow_rrt=phase == 'TRANSFER')
                        events.append(dict(time_s=t, kind='TRANSITION_REANCHOR', phase=phase,
                                           success=revised is not None, failure=failure))
                        if revised is None:
                            phase = 'SELECT' if np.any(safe) else 'LIFT_PLAN'
                            nominal = measured
                            fingers.rebase(nominal)
                            mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
                        else:
                            transfer = revised
                else:
                    phase = 'REACQUIRE'; phase_started = t; stable_since = None; offset = 0.
                    nominal = transfer.waypoints[-1].copy()
                    nominal[7:] = joint_state(h, data)[7:]
                    reacquire_nominal = nominal.copy()
                    fingers.rebase(nominal)
                    mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
        elif phase == 'REACQUIRE':
            nominal = reacquire_nominal.copy()
            if np.any(safe):
                stable_since = t if stable_since is None else stable_since
            else:
                stable_since = None
                if not in_relief:
                    offset = min(.006, offset+.0015*.002)
            nominal[:7] += endpoint_arm_offset(h, data, normal, offset)
            if stable_since is not None and t-stable_since >= .15:
                events.append(dict(time_s=t, kind='LIFT_RECONTACT', approach_offset_m=offset))
                phase = 'SELECT'
            elif t-phase_started > 5.:
                events.append(dict(time_s=t, kind='REACQUIRE_TIMEOUT'))
                phase = 'LIFT_PLAN'
        # Intentional withdrawal must not fight an inward force controller.
        search_enabled = phase in ('SLIDE', 'LOCAL_SEARCH', 'NEAR_TRANSFER', 'APPROACH', 'REACQUIRE') and not in_relief
        if coordinated is not None and phase not in ('FINGER_RECOVERY', 'FINGER_STABILIZE'):
            events.append(dict(time_s=t, kind='COORDINATED_RECOVERY_END', sequence=recovery_sequence,
                status='ABORTED', reason='OUTER_FORCE_OR_PHASE_REPLAN', selected_finger=coordinated.path.selected_finger,
                measured_recovery_success=False))
            coordinated = None
            coordinated_gate = None
        search_mask = np.ones(4, dtype=bool)
        if coordinated is not None and phase in ('FINGER_RECOVERY', 'FINGER_STABILIZE'):
            search_enabled = bool(coordinated_result['search_enabled']) and not in_relief
            # Preserve the selected finger's planned configuration through a
            # pivot. A local search would invalidate its contact-anchored path.
            if coordinated.kind == 'WRIST_PIVOT' and coordinated.contact_pose is None:
                search_enabled = False
            search_mask[:] = False
            search_mask[coordinated.path.selected_finger] = True
        # The bounded solver must see the reference actually sent to the
        # actuators, including existing preload. Adding it after solving would
        # invalidate its joint/bias constraints near a limit.
        if phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'):
            mcc.reset()
        handoff_bias = np.r_[np.zeros(7), mcc.held_bias_rad]
        search_nominal = nominal+handoff_bias
        finger_nominal, finger_diag = fingers.update(data, search_nominal, forces, normals, active,
            enabled=search_enabled, search_mask=search_mask,
            intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
        if any(finger_diag['requires_rebase']):
            events.append(dict(time_s=t, kind='FINGER_REFERENCE_REBASE_REQUIRED',
                fingers=np.flatnonzero(finger_diag['requires_rebase']).tolist(), phase=phase,
                solver=finger_diag['solver']))
            if coordinated is not None:
                events.append(dict(time_s=t, kind='COORDINATED_RECOVERY_END', sequence=recovery_sequence,
                    status='ABORTED', reason='FINGER_REFERENCE_REBASE_REQUIRED',
                    selected_finger=coordinated.path.selected_finger, measured_recovery_success=False))
                coordinated = None
                coordinated_gate = None
            nominal = joint_state(h, data)
            fingers.rebase(nominal)
            mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
            # Keep this tick's explicit held proposal; next tick replans from
            # measurement. The common final command bounds still apply.
            phase = 'SELECT' if np.any(safe) else 'LOCAL_SEARCH'
            phase_started = t
        # Observe every native tick: a 2 ms physical gap must not be hidden by
        # the 10 ms report sampling when deciding a 150 ms stable recontact.
        participation_diag = participation.update(t, joint_state(h, data),
            data.geom_xpos[h.tip_geom_ids].copy(), active, 'FORCE_RELIEF' if in_relief else phase, forces_n=forces,
            planned_targets_world_m=finger_diag['target_positions_world_m'],
            target_valid=finger_diag['target_valid'],
            target_available=np.linalg.norm(finger_diag['target_normals_world'], axis=1) > .5,
            search_state=finger_diag['state'], search_delta_rad=finger_diag['search_delta_rad'],
            request_allowed_mask=~np.asarray(recovery_queue.pending_mask, dtype=bool))
        context_diag = recovery_context.update(t, pose(h, data), active, forces,
                                              'FORCE_RELIEF' if in_relief else phase)
        context_native_window.append(dict(time_s=t, native_time_s=float(data.time),
            q_rad=joint_state(h, data),
            qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs], data.qvel[h.hand_dof_adrs]],
            sensor_command_rad=last.copy(), qacc_warmstart=data.qacc_warmstart.copy(),
            palm_pose_world=pose(h, data), active=active.copy(), forces_n=forces.copy(),
            phase='FORCE_RELIEF' if in_relief else phase))
        events.extend(context_diag['events'])
        if participation_diag['request'] is not None:
            emitted_request = participation_diag['request']
            noted_request = recovery_context.note_request(int(emitted_request['finger']), t)
            if not noted_request['accepted']:
                events.append(dict(time_s=t, kind='RECOVERY_CONTEXT_REQUEST_INCONSISTENCY',
                                   request=emitted_request, observer_result=noted_request))
                (directory/'events.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
                raise RuntimeError('Context observer rejected a current emitted monitor request')
            events.append(noted_request)
            events.append(dict(time_s=t, **{k:v for k,v in emitted_request.items() if k != 'time_s'}))
        # A distant new measured wrist pose can make the old failed local
        # search relevant again. Renew only exhausted finite quotas, while a
        # real safe finger supports the hand. This neither moves the plant nor
        # resets the unresolved physical loss or regional coverage clocks.
        if coordinated is None and not in_relief and phase in ('SLIDE', 'LOCAL_SEARCH', 'REACQUIRE'):
            for finger in range(4):
                exhausted = (fingers.search_elapsed[finger] >= fingers.config.maximum_search_time_s
                    or fingers.search_distance[finger] >= fingers.config.maximum_search_distance_m-1e-12
                    or participation_diag['per_finger'][finger]['request_budget_remaining'] <= 0
                    or fingers.posture_attempts_this_context[finger] >= fingers.maximum_posture_attempts_per_loss_episode)
                if active[finger] or not exhausted:
                    continue
                if (fingers.was_active[finger] or participation._loss_start[finger] is None
                    or fingers.recovery_context_id[finger] != participation._recovery_context_id[finger]
                    or fingers.recovery_context_id[finger] != context_diag['per_finger'][finger]['context_id']):
                    raise RuntimeError('Context consumers are not synchronized to the current measured loss')
                budget_before = dict(search_elapsed_s=float(fingers.search_elapsed[finger]),
                    search_distance_m=float(fingers.search_distance[finger]),
                    attempts_this_context=int(fingers.posture_attempts_this_context[finger]),
                    request_budget_remaining=participation_diag['per_finger'][finger]['request_budget_remaining'])
                grant = recovery_context.grant_new_context(finger, t, budget_exhausted=True)
                if grant['accepted']:
                    search_grant = fingers.apply_recovery_context(grant)
                    monitor_grant = participation.apply_recovery_context(grant)
                    if not search_grant['accepted'] or not monitor_grant['accepted']:
                        raise RuntimeError('Measured context consumers rejected the same current observation')
                    window_name = f'recovery_context_{finger}_{grant["context_id"]:04d}_native.npz'
                    np.savez_compressed(directory/window_name,
                        **{key: np.asarray([row[key] for row in context_native_window])
                           for key in context_native_window[0]})
                    events.append(dict(**grant, search_consumer=search_grant,
                                       participation_consumer=monitor_grant,
                                       phase=phase, active=active.tolist(), forces_n=forces.tolist(),
                                       budget_before=budget_before,
                                       native_support_window=window_name,
                                       native_support_window_sha256=sha256((directory/window_name).read_bytes()).hexdigest(),
                                       budget_effect='NEXT_NATIVE_TICK', measured_recovery_success=False))
        endpoint_recovery.update(t, active, forces,
            [int(item['episode_id']) for item in context_diag['per_finger']],
            fingers.recovery_context_id, 'FORCE_RELIEF' if in_relief else phase, pose(h, data))
        queue_diag = recovery_queue.update(t, active, forces,
            [int(item['episode_id']) for item in context_diag['per_finger']],
            fingers.recovery_context_id, 'FORCE_RELIEF' if in_relief else phase)
        events.extend(queue_diag['drop_events'])
        if participation_diag['request'] is not None:
            emitted_request = participation_diag['request']
            selected = int(emitted_request['finger'])
            queued = recovery_queue.enqueue(emitted_request,
                episode_id=int(context_diag['per_finger'][selected]['episode_id']),
                context_id=int(fingers.recovery_context_id[selected]))
            if not queued['accepted']:
                raise RuntimeError('Pending-request mask failed to prevent a stale/duplicate enqueue')
            events.append(queued)
        if coordinated is None and not in_relief and np.any(safe):
            dispatched = recovery_queue.take_boundary(t, natural_stroke_endpoint=natural_endpoint_this_tick)
            if dispatched['accepted']:
                if dispatched['mode'] == 'SHORT_FIXED_WRIST':
                    noted_endpoint = endpoint_recovery.note_boundary_dispatch(dispatched)
                    if not noted_endpoint['accepted']:
                        raise RuntimeError('Endpoint policy rejected a current natural-boundary dispatch')
                events.append(dict(**dispatched, actual_native_state=dict(
                    q_rad=joint_state(h, data).tolist(),
                    qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs], data.qvel[h.hand_dof_adrs]].tolist(),
                    sensor_command_rad=last.tolist(), qacc_warmstart=data.qacc_warmstart.tolist(),
                    native_time_s=float(data.time), active=active.tolist(), forces_n=forces.tolist(),
                    episode_ids=[int(item['episode_id']) for item in context_diag['per_finger']],
                    context_ids=fingers.recovery_context_id.tolist())))
                coordinated_request = dict(dispatched['request'], dispatch_mode=dispatched['mode'],
                    dispatch_context_id=dispatched['context_id'], dispatch_episode_id=dispatched['episode_id'])
        # Search has already incorporated the preload. Remove it from this
        # argument because ContactPreservingMCC itself adds it exactly once.
        command = (finger_nominal.copy() if phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN') else
                   mcc.command(data, finger_nominal-np.r_[np.zeros(7), mcc.held_bias_rad],
                               forces, normals, active, 2.))
        if in_relief:
            command = applied_reference_force_release(h, data, command, last, forces, normals, config)
            command[:7] = arm_force_release(h, data, last, forces, normals)
        command = np.clip(command, last-.0024, last+.0024)
        command = np.clip(command, planner.lower, planner.upper)
        if step % 5 == 0:
            reason = planner.command_collision_reason(joint_state(h, data), command)
            if reason in ('NON_TIP_CLEARANCE', 'NON_TIP_OBJECT_COLLISION', 'SELF_COLLISION',
                          'ARM_SELF_CLEARANCE', 'SHALLOW_ARM_SELF_COLLISION'):
                events.append(dict(time_s=t, kind='COMMAND_GEOMETRY_DEFER', reason=reason, phase=phase))
                command = np.clip(joint_state(h, data), last-.0024, last+.0024)
                command = np.clip(command, planner.lower, planner.upper)
                phase = ('SELECT' if np.any(safe) else 'LIFT_PLAN') if reason in (
                    'ARM_SELF_CLEARANCE', 'SHALLOW_ARM_SELF_COLLISION') else 'LIFT_PLAN'
                nominal = joint_state(h, data)
                fingers.rebase(nominal); mcc = ContactPreservingMCC(h, .002, nominal, last, active, forces,
                intentional_air=phase in ('LIFT', 'TRANSFER', 'LIFT_PLAN'))
            values = dict(time_s=t, q_rad=joint_state(h, data),
                qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs], data.qvel[h.hand_dof_adrs]],
                command_rad=command.copy(), sensor_command_rad=last.copy(),
                palm_pose_world=pose(h, data), tip_positions_world_m=tip_positions(h, data),
                contact_positions_world_m=points.copy(), normals_world=normals.copy(), forces_n=forces.copy(),
                active=active.copy(), coverage=coverage.area_fraction,
                covered_goals=int(coverage.visited_goals.sum()), goal_id=-1,
                phase='FORCE_RELIEF' if in_relief else observed_phase,
                command_phase=phase, endpoint_offset_m=offset,
                stroke_id=stroke_id, stroke_distance_m=s)
            values['arm_tracking_error_rad'] = float(np.max(abs(joint_state(h, data)[:7]-last[:7])))
            values['arm_self_force_n'] = self_force
            values['finger_search_active'] = np.asarray(finger_diag['seeking'], dtype=bool)
            values['finger_search_enabled'] = bool(search_enabled)
            values['finger_search_bias_rad'] = np.asarray(finger_diag['joint_bias_rad']).copy()
            values['finger_search_distance_m'] = np.asarray(finger_diag['search_distance_m']).copy()
            values['finger_search_delta_rad'] = np.asarray(finger_diag['search_delta_rad']).copy()
            values['finger_search_target_positions_world_m'] = np.asarray(finger_diag['target_positions_world_m']).copy()
            values['finger_search_target_normals_world'] = np.asarray(finger_diag['target_normals_world']).copy()
            values['finger_search_target_valid'] = np.asarray(finger_diag['target_valid'], dtype=bool)
            values['finger_search_state'] = np.asarray(finger_diag['state'])
            values['finger_search_loss_age_s'] = np.asarray(finger_diag['loss_age_s']).copy()
            values['finger_search_elapsed_s'] = np.asarray(finger_diag['search_elapsed_s']).copy()
            values['finger_contact_stable_age_s'] = np.asarray(finger_diag['contact_stable_age_s']).copy()
            values['finger_posture_attempt_id'] = np.asarray(finger_diag['posture_attempt_id']).copy()
            values['finger_posture_attempts_this_episode'] = np.asarray(finger_diag['posture_attempts_this_episode']).copy()
            values['finger_recovery_context_id'] = fingers.recovery_context_id.copy()
            values['finger_posture_attempts_this_context'] = fingers.posture_attempts_this_context.copy()
            values['finger_search_next_elapsed_s'] = fingers.search_elapsed.copy()
            values['finger_search_next_distance_m'] = fingers.search_distance.copy()
            values['finger_search_measured_active'] = active.copy()
            values['mcc_handoff_bias_rad'] = mcc.held_bias_rad
            values['finger_search_nominal_rad'] = search_nominal.copy()
            values['finger_nominal_rad'] = finger_nominal.copy()
            values['pad_centers_world_m'] = data.geom_xpos[h.tip_geom_ids].copy()
            values['finger_loss_episode_age_s'] = np.asarray([
                item['unresolved_loss_episode_age_s'] for item in participation_diag['per_finger']])
            values['finger_recovery_requests'] = np.asarray([
                item['requests_issued_this_loss_episode'] for item in participation_diag['per_finger']])
            values['finger_recovery_context_requests'] = participation._context_request_count.copy()
            values['finger_recovery_pending'] = np.asarray(recovery_queue.pending_mask, dtype=bool)
            values['coordinated_recovery_sequence'] = recovery_sequence
            values['coordinated_recovery_stage'] = coordinated_stage
            values['coordinated_total_attempt_elapsed_s'] = (0. if coordinated is None else t-coordinated_attempt_started)
            values['coordinated_pivot_endpoint_age_s'] = (0. if coordinated_gate_diag is None
                else coordinated_gate_diag['endpoint_age_s'])
            values['coordinated_pivot_tracking_stable_s'] = (0. if coordinated_gate_diag is None
                else coordinated_gate_diag['stable_tracking_age_s'])
            values['coordinated_recovery_finger'] = -1 if coordinated is None else coordinated.path.selected_finger
            values['coordinated_path_elapsed_s'] = 0. if coordinated is None else coordinated.elapsed_s
            values['coordinated_status'] = 'INACTIVE' if coordinated is None else coordinated.status
            values['coordinated_anchor_safe_loss_s'] = (0. if coordinated is None or coordinated_result is None
                else coordinated_result['anchor_safe_loss_age_s'])
            values['nominal_path_distance_m'] = s
            values['qacc_warmstart'] = data.qacc_warmstart.copy()
            for key, value in values.items():
                logs.setdefault(key, []).append(value)
        if local_progress.open_stroke is not None and phase != 'SLIDE':
            ended = local_progress.finish(measured_count, phase)
            events.append(dict(time_s=t, kind='LOCAL_STROKE_RESULT', **ended))
        last = command.copy(); set_control(h, data, command)
        if step % 5000 == 0:
            progress = dict(time_s=t, phase=phase, stroke_id=stroke_id, coverage=coverage.area_fraction,
                            forces_n=forces.tolist(), peak_force_n=peak, wall_s=perf_counter()-started)
            print('SLIDE', json.dumps(progress), flush=True)
            (directory/'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
            (directory/'events.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
        if step % 30000 == 0:
            checkpoint(t)
        if full_since is not None and t-full_since >= .16 and step % 5 == 0:
            stop = 'FULL_MEASURED_SURFACE'; break
        if step % 500 == 0 and (directory/'stop_requested.json').exists():
            stop = 'DEVELOPMENT_STOP_REQUESTED'
            events.append(dict(time_s=t, kind='DEVELOPMENT_STOP_REQUESTED'))
            break
        if step == int(budget_s/.002):
            break
        mujoco.mj_step(h.model, data); mujoco.mj_forward(h.model, data)
    arrays = {key: np.asarray(value) for key, value in logs.items()}
    arrays['covered_vertices'] = coverage.visited_vertices
    arrays['required_goal_visited'] = coverage.visited_goals
    np.savez_compressed(directory/'trace.npz', **arrays)
    np.savez_compressed(directory/'final_state.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl,
                        qacc_warmstart=data.qacc_warmstart, time=data.time)
    (directory/'events.json').write_text(json.dumps(events, indent=2), encoding='utf-8')
    summary = dict(cell=cell, seed=seed, stop_reason=stop, executed_s=t, peak_force_n=peak,
        peak_arm_self_force_n=peak_arm_self_force,
        wall_s=perf_counter()-started, final_covered_area_fraction=coverage.area_fraction,
        covered_vertex_count=int(coverage.visited_vertices.sum()), vertex_count=len(graph.vertices_m),
        visited_required_goals=int(coverage.visited_goals.sum()), required_goal_count=graph.required_goal_count,
        full_surface_complete=complete_surface(coverage) and stop == 'FULL_MEASURED_SURFACE',
        completion_stability_s=.15, all_surface_measured=complete_surface(coverage), stroke_count=stroke_id+1,
        lift_count=sum(e['kind'] == 'LIFT_COMMIT' for e in events),
        trace_sha256=sha256((directory/'trace.npz').read_bytes()).hexdigest())
    summary['sliding_metrics'] = analyze_trace(arrays)
    (directory/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('SLIDE_RESULT', json.dumps({k:v for k,v in summary.items() if k != 'sliding_metrics'}), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cell', choices=['A-MCC', 'B-MCC'], default='B-MCC')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--budget', type=float, default=120.)
    parser.add_argument('--speed', type=float, default=.006)
    parser.add_argument('--expected-runtime-sha256')
    parser.add_argument('--witness-cache', type=Path)
    args = parser.parse_args()
    run(args.output, cell=args.cell, seed=args.seed, budget_s=args.budget, speed=args.speed,
        expected_runtime_sha256=args.expected_runtime_sha256, witness_path=args.witness_cache)
