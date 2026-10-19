"""Matched V3/V4 local geometry experiment from one real native state.

Only the search geometry differs. Plant state is exact; controller counters and
bias are restored where logged, while MCC history and target refresh age share
fresh initialization. This is not exact full-controller continuation, a new
whole traversal, or authority to award coverage.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import mujoco
import numpy as np
from .active_fingers_v3 import ActiveFingerSearchV3
from .contact_handoff import ContactPreservingMCC
from .contracts import ExperimentConfig
from .physics import contact_measurement,joint_state,pose,tip_positions,set_control
from .posture_recovery_probe import prepare_source,make_plant,snapshot_sources
from .recovery import SharedContactRecovery
from .sliding_safety import SlidingSafetyPlanner,arm_self_contacts

DT=.002
RESTORED={
    'finger_search_bias_rad':'bias','finger_search_distance_m':'search_distance',
    'finger_search_elapsed_s':'search_elapsed','finger_search_loss_age_s':'loss_age',
    'finger_contact_stable_age_s':'contact_stable_age','finger_posture_attempt_id':'posture_attempt_id',
    'finger_posture_attempts_this_episode':'posture_attempts_this_episode',
    'finger_posture_attempts_this_context':'posture_attempts_this_context',
    'finger_recovery_context_id':'recovery_context_id'}
GUARDED_COMMAND_REASONS=('NON_TIP_CLEARANCE','NON_TIP_OBJECT_COLLISION','SELF_COLLISION',
                        'ARM_SELF_CLEARANCE','SHALLOW_ARM_SELF_COLLISION')


def digest(value):return sha256(value).hexdigest()


def run(directory,source,controller_frame,search_type,*,duration_s,sources):
    directory=Path(directory)
    if directory.exists():raise FileExistsError('Use new evidence directories')
    directory.mkdir(parents=True)
    started=perf_counter();h,graph,data,sensor_errors=make_plant(source)
    finger=int(source['provenance']['selected_finger']);columns=7+4*finger+np.arange(4)
    ids=np.r_[h.arm_actuator_ids,h.hand_actuator_ids]
    last=data.ctrl[ids].copy();initial_ctrl=last.copy()
    initial={key:np.asarray(getattr(data,key)).copy() for key in ('qpos','qvel','ctrl','qacc_warmstart','time')}
    forces,points,normals,active,_,_=contact_measurement(h,data,.2)
    search=search_type(h,graph);search.rebase(last)
    for key,attribute in RESTORED.items():setattr(search,attribute,controller_frame[key].copy())
    search.was_active[:]=active
    search.stable_confirmed[:]=active&(forces<8.)&(search.contact_stable_age>=.15)
    # Existing servo preload is already in priorctrl. Subtract restored search
    # bias exactly once so initial effective search reference remains priorctrl.
    nominal=last-np.r_[np.zeros(7),search.bias]
    mcc=ContactPreservingMCC(h,DT,last,last,active,forces)
    cfg=ExperimentConfig(seed=source['seed']);guard=SlidingSafetyPlanner(h,cfg);relief=SharedContactRecovery(cfg)
    object_geoms=[int(g) for g in h.object_geom_ids if h.model.geom_contype[g]!=0]
    selected_geom=int(h.tip_geom_ids[finger]);mask=np.zeros(4,bool);mask[finger]=True
    np.savez_compressed(directory/'initial.npz',**initial)
    np.savez_compressed(directory/'source_frame.npz',**source['frame'])
    np.savez_compressed(directory/'controller_frame.npz',**controller_frame)
    np.savez_compressed(directory/'controller_handoff.npz',effective_initial_reference=last,
        base_reference=nominal,restored_search_bias=search.bias,initial_mcc_held_bias=mcc.held_bias_rad)
    for name,raw in (('model.xml',source['model_bytes']),('surface_graph.npz',source['graph_bytes']),
                    ('source_run_manifest.json',source['source_manifest_bytes']),('source_clock_anchor.npz',source['clock_anchor_bytes'])):
        (directory/name).write_bytes(raw)
    for name,raw in sources.items():
        target=directory/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    manifest=dict(protocol='i04-support-point-search.paired.development.v1',source_state=source['provenance'],
        selected_finger=finger,search_implementation=search_type.__name__,dt_s=DT,observation_duration_s=duration_s,
        initial_sensor_errors=sensor_errors,source_hashes={k:digest(v) for k,v in sources.items()},
        model_sha256=digest(source['model_bytes']),object_position_m=source['object_position_m'],
        controller_initialization='SHARED_FRESH_MCC_AND_TARGET_AGE_WITH_RESTORED_LOGGED_COUNTERS_AND_SEARCH_BIAS',
        exact_full_controller_continuation=False,restored_observables=RESTORED,
        initial_reference='Exact prior actuator command; base=priorctrl-restored search bias; no double preload',
        arm_reference='Fixed initial prior control, identical in both cases',
        contact_handoff='First actual safe selected contact reanchors all effective references to current priorctrl; history/budgets survive V3.rebase',
        requires_rebase='Explicit probe stop; no silent path or budget reset',
        common_controls=dict(desired_force_n=2.,maximum_command_speed_rad_s=1.2,
            force_release_at_n=8.,release_until_below_n=4.,hard_force_n=12.,hard_force_duration_s=.020,
            arm_self_stop_n=.5,command_geometry_guard_reasons=GUARDED_COMMAND_REASONS,
            pad_preload_reference='DEEP_PAD command FK alone is not plant contact; original native force/collision guards apply',
            search_mask=mask.tolist(),mcc_all_actual_contacts=True,stable_confirmation_s=.15),
        active_finger_search=asdict(search.config),gap_sampling_s=.010,
        coverage_authority='NONE',coverage_counted=False,whole_traversal_evidence=False,
        controller_frame_sha256=digest((directory/'controller_frame.npz').read_bytes()),
        controller_handoff_sha256=digest((directory/'controller_handoff.npz').read_bytes()))
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2))
    events=[];logs={};peak=peak_arm=0.;high_steps=0;first=stable_since=confirmed=None
    latched=False;in_relief=False;stop='OBSERVATION_COMPLETE'
    for step in range(round(duration_s/DT)+1):
        t=step*DT;forces,points,normals,active,unsafe,penetration=contact_measurement(h,data,.2)
        arm=max([0.]+[c['force_n'] for c in arm_self_contacts(h,data,0.)])
        peak=max(peak,float(forces.max()));peak_arm=max(peak_arm,arm)
        high_steps=high_steps+1 if forces.max()>12. else 0
        hardware=('ARM_SELF_COLLISION' if arm>.5 else 'NON_TIP_OBJECT_COLLISION' if unsafe else
            'SUSTAINED_DANGEROUS_FORCE' if high_steps*DT>=.020 else
            'NONFINITE_STATE' if not np.isfinite(data.qpos).all() else
            'JOINT_LIMIT' if np.any(joint_state(h,data)<guard.lower-.02) or np.any(joint_state(h,data)>guard.upper+.02) else None)
        selected_safe=bool(active[finger] and .2<=forces[finger]<8.)
        if selected_safe:
            if first is None:first=t;events.append(dict(time_s=t,kind='ACTUAL_FIRST_RECONTACT',finger=finger))
            stable_since=t if stable_since is None else stable_since
            if confirmed is None and t-stable_since>=.15-1e-12:
                confirmed=t;events.append(dict(time_s=t,kind='ACTUAL_STABLE_RECONTACT',finger=finger,stable_s=t-stable_since))
        else:stable_since=None
        if selected_safe and not latched:
            nominal=last.copy();search.rebase(nominal)
            mcc=ContactPreservingMCC(h,DT,nominal,last,active,forces);latched=True
            events.append(dict(time_s=t,kind='PRIOR_CONTROL_CONTACT_HANDOFF',prior_control=last.tolist(),clear_loss_history=False))
        next_relief=bool(forces.max()>=8. or (in_relief and forces.max()>=4.))
        if next_relief!=in_relief:
            nominal=last.copy();search.rebase(nominal);mcc=ContactPreservingMCC(h,DT,nominal,last,active,forces)
            events.append(dict(time_s=t,kind='FORCE_RELEASE_BEGIN' if next_relief else 'FORCE_RELEASE_END'))
        in_relief=next_relief
        search_nominal=nominal+np.r_[np.zeros(7),mcc.held_bias_rad]
        reference,diag=search.update(data,search_nominal,forces,normals,active,enabled=not in_relief,search_mask=mask)
        # Only selected search proposal is connected; other references retain
        # their already applied preload and actual-force MCC.
        proposed=search_nominal+np.r_[np.zeros(7),search.bias]
        proposed[columns]=reference[columns]
        command=mcc.command(data,proposed-np.r_[np.zeros(7),mcc.held_bias_rad],forces,normals,active,2.)
        if in_relief:command=relief.relieve_force(h,data,command,forces,normals)
        command=np.clip(np.clip(command,last-.0024,last+.0024),guard.lower,guard.upper)
        reason=guard.command_collision_reason(joint_state(h,data),command)
        rebase=bool(np.asarray(diag['requires_rebase'])[finger])
        terminal=bool(hardware or reason in GUARDED_COMMAND_REASONS or rebase or step==round(duration_s/DT))
        if hardware:stop=hardware
        elif reason in GUARDED_COMMAND_REASONS:stop='COMMAND_GEOMETRY_STOP'
        elif rebase:stop='SEARCH_REBASE_REQUIRED'
        if terminal:command=last.copy()
        gap=(min(mujoco.mj_geomDistance(h.model,data,selected_geom,g,.2,None) for g in object_geoms)
             if step%5==0 or terminal else np.nan)
        values=dict(time_s=t,native_time_s=float(data.time),q_rad=joint_state(h,data),
            qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs],data.qvel[h.hand_dof_adrs]],
            sensor_command_rad=last.copy(),command_rad=command.copy(),qacc_warmstart=data.qacc_warmstart.copy(),
            forces_n=forces,active=active,normals_world=normals,contact_positions_world_m=points,
            tip_positions_world_m=tip_positions(h,data),palm_pose_world=pose(h,data),
            arm_self_force_n=arm,non_tip_object_collision=bool(unsafe),maximum_penetration_m=penetration,
            phase='FORCE_RELEASE' if in_relief else 'FIXED_WRIST_CONTACT' if selected_safe else 'FIXED_WRIST_SEARCH',
            nominal_rad=nominal.copy(),search_nominal_rad=search_nominal.copy(),search_proposal_rad=proposed.copy(),
            mcc_held_bias_rad=mcc.held_bias_rad,command_geometry_reason=reason or '',
            measured_selected_collision_gap_m=gap,coverage=0.,covered_goals=0,
            finger_search_active=np.asarray(diag['seeking']),finger_search_delta_rad=np.asarray(diag['search_delta_rad']),
            finger_search_bias_rad=np.asarray(diag['joint_bias_rad']),finger_search_state=np.asarray(diag['state']),
            finger_search_loss_age_s=np.asarray(diag['loss_age_s']),finger_search_elapsed_s=np.asarray(diag['search_elapsed_s']),
            finger_search_distance_m=np.asarray(diag['search_distance_m']),finger_contact_stable_age_s=np.asarray(diag['contact_stable_age_s']),
            finger_search_target_positions_world_m=np.asarray(diag['target_positions_world_m']),
            finger_search_target_normals_world=np.asarray(diag['target_normals_world']),finger_search_target_valid=np.asarray(diag['target_valid']),
            finger_search_requires_rebase=np.asarray(diag['requires_rebase']),finger_posture_attempt_id=np.asarray(diag['posture_attempt_id']),
            finger_posture_attempts_this_episode=np.asarray(diag['posture_attempts_this_episode']),
            finger_recovery_context_id=np.asarray(diag['recovery_context_id']),
            finger_search_surface_points_world_m=np.asarray(diag.get('surface_points_world_m',np.full((4,3),np.nan))),
            finger_search_surface_face_ids=np.asarray(diag.get('surface_face_ids',np.full(4,-1))),
            finger_search_center_predicted_step_m=np.asarray([(x or {}).get('center_step_m',(x or {}).get('cartesian_step_m',0.)) for x in diag['solver']]),
            finger_search_support_predicted_step_m=np.asarray([(x or {}).get('support_step_m',np.nan) for x in diag['solver']]),
            finger_search_charged_step_m=np.asarray([(x or {}).get('charged_step_m',(x or {}).get('cartesian_step_m',0.)) for x in diag['solver']]))
        for key,value in values.items():logs.setdefault(key,[]).append(value)
        if terminal:
            if stop!='OBSERVATION_COMPLETE':events.append(dict(time_s=t,kind='PROBE_STOP',reason=stop,command_reason=reason,solver=diag['solver'][finger]))
            break
        last=command.copy();set_control(h,data,command);mujoco.mj_step(h.model,data);mujoco.mj_forward(h.model,data)
    arrays={key:np.asarray(value) for key,value in logs.items()}
    np.savez_compressed(directory/'trace.npz',**arrays)
    np.savez_compressed(directory/'final_state.npz',**{key:np.asarray(getattr(data,key)).copy() for key in ('qpos','qvel','ctrl','qacc_warmstart','time')})
    (directory/'events.json').write_text(json.dumps(events,indent=2))
    summary=dict(protocol=manifest['protocol'],search_implementation=search_type.__name__,selected_finger=finger,
        stop_reason=stop,executed_s=t,native_recorded_states=len(arrays['time_s']),first_recontact_delay_s=first,
        stable_confirmation_delay_s=confirmed,stable_recontact_observed=confirmed is not None,
        success=confirmed is not None and stop=='OBSERVATION_COMPLETE',peak_force_n=peak,peak_arm_self_force_n=peak_arm,
        final_selected_safe_contact=selected_safe,final_selected_force_n=float(forces[finger]),
        initial_collision_gap_m=float(arrays['measured_selected_collision_gap_m'][0]),final_collision_gap_m=float(gap),
        arm_command_change_rad=float(np.max(abs(arrays['command_rad'][:,:7]-initial_ctrl[:7]))),
        initial_state_restore_count=1,subsequent_state_restores=0,coverage_counted=False,whole_traversal_evidence=False,
        trace_sha256=digest((directory/'trace.npz').read_bytes()),wall_s=perf_counter()-started)
    (directory/'summary.json').write_text(json.dumps(summary,indent=2));print('SUPPORT_SEARCH_PROBE',json.dumps(summary),flush=True)
    return summary


def main():
    from .active_fingers_v4 import ActiveFingerSearchV4
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--source-time',type=float,default=428.64)
    parser.add_argument('--finger',type=int,default=1);parser.add_argument('--duration',type=float,default=10.)
    args=parser.parse_args();source=prepare_source(args.source,args.source_time,args.finger)
    with np.load(source['provenance']['source_trace']['path']) as archive:
        row=source['provenance']['source_trace']['row'];controller_frame={k:archive[k][row].copy() for k in RESTORED}
    sources=snapshot_sources();root=Path(__file__).resolve().parents[2]
    sources[Path(__file__).relative_to(root).as_posix()]=Path(__file__).read_bytes()
    results={name:run(args.output/name,source,controller_frame,cls,duration_s=args.duration,sources=sources)
             for name,cls in (('V3',ActiveFingerSearchV3),('V4',ActiveFingerSearchV4))}
    for name in ('initial.npz','controller_frame.npz','controller_handoff.npz'):
        with np.load(args.output/'V3'/name) as a,np.load(args.output/'V4'/name) as b:
            assert set(a.files)==set(b.files)
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    result=dict(scope='MATCHED_LOCAL_SUPPORT_GEOMETRY_ABLATION_NOT_EXACT_CONTROLLER_CONTINUATION',
        identical_initial_physics_and_observable_controller_state=True,coverage_counted=False,cases=results)
    (args.output/'paired_summary.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
