"""Independent native continuation from a stopped traversal, never fresh success.

Recover the final solver state by replaying the last two real 2 ms controls.
Execute a separately checked retreat/hand-preparation/transfer candidate and
measure any new coverage in this probe's own ledger. The old run is read-only.
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

from .active_fingers import ActiveFingerSearch
from .contracts import ExperimentConfig
from .full_surface import JointPath, endpoint_arm_offset
from .oracle import MeasuredCoverage
from .physics import contact_measurement,joint_state,pose,set_control,tip_positions
from .posture_recovery_probe import prepare_source,make_plant,snapshot_sources
from .recovery import SharedContactRecovery
from .sliding_safety import SlidingSafetyPlanner,arm_self_contacts
from .surface_slide import GraceMCC,arm_force_release,tracking_progress_scale

DT=.002


def _hash(path):return sha256(Path(path).read_bytes()).hexdigest()


def restore_terminal(run):
    """Restore once then integrate exactly two native ticks to the final state."""
    with np.load(run/'trace.npz') as tr:
        last_time=float(tr['time_s'][-1]);first_control=tr['command_rad'][-1].copy()
    source=prepare_source(run,last_time,2)
    h,graph,data,errors=make_plant(source)
    with np.load(run/'final_state.npz') as z: final={k:z[k].copy() for k in z.files}
    assert abs(float(final['time'])-float(data.time)-2*DT)<1e-10
    start=dict(qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy(),
        qacc_warmstart=data.qacc_warmstart.copy(),time=float(data.time))
    set_control(h,data,first_control)
    mujoco.mj_step(h.model,data);mujoco.mj_forward(h.model,data)
    # This actuator-order command is the exact second control saved by the
    # old run; no inference or qpos assignment is used during replay.
    data.ctrl[:]=final['ctrl']
    mujoco.mj_step(h.model,data);mujoco.mj_forward(h.model,data)
    replay_errors={k:float(np.max(abs(np.asarray(getattr(data,k))-final[k])))
        for k in ('qpos','qvel','ctrl','time')}
    assert max(replay_errors.values())<1e-10,replay_errors
    return h,graph,data,source,start,first_control,replay_errors,errors


def restore_continuation(run):
    """Continue a recorded real probe endpoint with exact native state/clock."""
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if not manifest['protocol'].startswith('i04-terminal-route-continuation'):
        return restore_terminal(run)
    with np.load(run/'trace.npz') as tr:
        keys=('q_rad','qvel_rad_s','sensor_command_rad','qacc_warmstart','forces_n','normals_world',
              'active','contact_positions_world_m','tip_positions_world_m','palm_pose_world')
        frame={key:tr[key][-1].copy() for key in keys}
        clock=float(tr['native_time_s'][-1]);relative=float(tr['time_s'][-1])
    source=dict(frame=frame,model_bytes=(run/'model.xml').read_bytes(),graph_bytes=(run/'surface_graph.npz').read_bytes(),
        object_position_m=manifest['object_position_m'],provenance=dict(native_time_s=clock,
            source_trace=dict(path=str(run/'trace.npz'),sha256=_hash(run/'trace.npz'),relative_time_s=relative),
            parent_manifest_sha256=_hash(run/'manifest.json'),parent_final_state_sha256=_hash(run/'final_state.npz')))
    h,graph,data,errors=make_plant(source)
    with np.load(run/'final_state.npz') as z:final={key:z[key].copy() for key in z.files}
    exact={key:float(np.max(abs(np.asarray(getattr(data,key))-final[key])))
           for key in ('qpos','qvel','ctrl','time','qacc_warmstart')}
    assert max(exact.values())<1e-10,exact
    initial={key:np.asarray(getattr(data,key)).copy() for key in ('qpos','qvel','ctrl','time','qacc_warmstart')}
    return h,graph,data,source,initial,None,exact,errors


def run_probe(directory,run_directory,route_path,*,duration_s=70.):
    directory=Path(directory).resolve();run_directory=Path(run_directory).resolve();route_path=Path(route_path).resolve()
    if directory.exists():raise FileExistsError('Use a fresh probe output')
    directory.mkdir(parents=True)
    h,graph,data,source,replay_start,replay_control,replay_errors,source_errors=restore_continuation(run_directory)
    source_files=snapshot_sources();own=Path(__file__).resolve();repo=own.parents[2]
    source_files[own.relative_to(repo).as_posix()]=own.read_bytes()
    for name,raw in source_files.items():
        dest=directory/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    for name in ('model.xml','surface_graph.npz','final_state.npz','summary.json','manifest.json'):
        (directory/('source_'+name if name not in ('model.xml','surface_graph.npz') else name)).write_bytes((run_directory/name).read_bytes())
    route_bytes=route_path.read_bytes();(directory/'route.npz').write_bytes(route_bytes)
    with np.load(route_path) as z:route={k:z[k].copy() for k in z.files}
    np.testing.assert_array_equal(joint_state(h,data),route['q'][0])
    np.savez_compressed(directory/'replay_start.npz',**replay_start,
        **({} if replay_control is None else dict(first_control_canonical=replay_control)))
    np.savez_compressed(directory/'initial.npz',qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl,
                        qacc_warmstart=data.qacc_warmstart,time=data.time)
    initial_time=float(data.time)
    config=ExperimentConfig(duration_s=duration_s+5,wrist_speed_m_s=.025,
        wrist_rotation_speed_rad_s=.35,trajectory_joint_speed_rad_s=.8)
    planner=SlidingSafetyPlanner(h,config)
    path=JointPath(route['q'],route['durations'],pose(h,data),np.zeros(7),0.)
    coverage=MeasuredCoverage(graph,config)
    coverage_source=run_directory/'coverage_independent.npz'
    if not coverage_source.exists():coverage_source=run_directory/'trace.npz'
    with np.load(coverage_source) as z:initial_mask=z['covered_vertices'].copy().astype(bool)
    np.savez_compressed(directory/'source_coverage.npz',covered_vertices=initial_mask)
    coverage.visited_vertices[:]=initial_mask;coverage.visited_goals[:]=initial_mask[graph.required_vertices]
    missing=np.flatnonzero(~initial_mask)
    goal_vertex=int(route['goal_vertex']);normal=graph.normals[goal_vertex].copy()
    mcc=GraceMCC(h,DT);fingers=ActiveFingerSearch(h,graph,dt=DT);relief=SharedContactRecovery(config)
    nominal=path.root_q.copy();fingers.rebase(nominal)
    last=np.r_[data.ctrl[h.arm_actuator_ids],data.ctrl[h.hand_actuator_ids]]
    manifest=dict(protocol='i04-terminal-route-continuation.development.v2',scope='INDEPENDENT_TERMINAL_CONTINUATION_NOT_FRESH_FULL_RUN',
        source_run=str(run_directory),source_runtime_protocol=json.loads((run_directory/'manifest.json').read_text())['protocol'],
        source_final_state_sha256=_hash(run_directory/'final_state.npz'),source_trace_sha256=source['provenance']['source_trace']['sha256'],
        source_coverage_sha256=_hash(coverage_source),source_provenance=source['provenance'],
        exact_terminal_replay_or_chain_state_errors=replay_errors,source_sensor_reconstruction_errors=source_errors,
        native_initial_time_s=initial_time,route_sha256=sha256(route_bytes).hexdigest(),planned_duration_s=path.duration_s,
        initial_uncovered_vertices=missing.tolist(),initial_covered_vertices=int(initial_mask.sum()),goal_vertex=goal_vertex,
        initial_covered_area_fraction=coverage.area_fraction,config=asdict(config),object_position_m=h.object_position_m.tolist(),
        original_search=asdict(fingers.config),measured_coverage='Original unchanged 10ms actual-contact geodesic rule; separate copied ledger',
        control='Original MCC disabled for intentional air, original active-finger search and GraceMCC on reacquire; original soft/hard guards',
        source_hashes={name:sha256(raw).hexdigest() for name,raw in source_files.items()})
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    logs={};events=[];cursor=0.;offset=0.;phase='LIFT';started=perf_counter();phase_started=0.
    peak=0.;peak_arm=0.;high_run=0;stable_since=None;goal_stable_since=None;goal_confirm=None
    first_goal_touch=None;stop='OBSERVATION_COMPLETE';relief_active=False;settle_since=None
    ends=np.cumsum(path.segment_durations)
    phases=route.get('segment_phases',np.array(['LIFT']*3+['HAND_PREPARATION']+['TRANSFER']*(len(ends)-4)))
    assert len(phases)==len(ends)
    for step in range(int(duration_s/DT)+1):
        t=step*DT;observed_phase=phase;measured=joint_state(h,data)
        forces,points,normals,active,unsafe,penetration=contact_measurement(h,data,.2)
        safe=active&(forces<=8.)
        self_contacts=arm_self_contacts(h,data,0.);self_force=max([0.]+[r['force_n'] for r in self_contacts])
        peak=max(peak,float(forces.max()));peak_arm=max(peak_arm,self_force)
        high_run=high_run+1 if forces.max()>12 else 0
        hard=('SHALLOW_ARM_SELF_COLLISION' if self_force>.5 else 'NON_TIP_OBJECT_COLLISION' if unsafe else
              'SUSTAINED_DANGEROUS_FORCE' if high_run*DT>=.02 else 'NONFINITE_STATE' if not np.all(np.isfinite(data.qpos)) else
              'JOINT_LIMIT' if np.any(measured<planner.lower-.02) or np.any(measured>planner.upper+.02) else None)
        if step%5==0:coverage.update(points-h.object_position_m,normals,active)
        near_goal=False
        safe_indices=np.flatnonzero(safe)
        if len(safe_indices):
            contact_vertices,residuals=graph.oriented_nearest_vertices(points[safe_indices]-h.object_position_m,normals[safe_indices])
            for finger,vertex,residual in zip(safe_indices,contact_vertices,residuals):
                if (residual<=config.mesh_residual_limit_m and graph.normals[vertex]@normals[finger]>=config.normal_alignment_cos
                        and goal_vertex in coverage.neighborhood(int(vertex))):
                    near_goal=True
        if near_goal:
            first_goal_touch=t if first_goal_touch is None else first_goal_touch
            goal_stable_since=t if goal_stable_since is None else goal_stable_since
            if goal_confirm is None and t-goal_stable_since>=.15:
                goal_confirm=t;events.append(dict(time_s=t,kind='TARGET_REGION_STABLE_CONTACT',vertex=goal_vertex))
        else:goal_stable_since=None
        previous_relief=relief_active;relief_active=bool(forces.max()>=8 or (relief_active and forces.max()>=4))
        relief.mode='FORCE_RELEASE' if relief_active else 'NORMAL'
        if relief_active and not previous_relief:
            events.append(dict(time_s=t,kind='FORCE_RELEASE_BEGIN',phase=phase,forces_n=forces.tolist()))
            mcc=GraceMCC(h,DT);phase='RELEASE_AND_ABORT'
        elif previous_relief and not relief_active:
            stop='ROUTE_ABORTED_AFTER_FORCE_RELEASE';hard=stop
        if hard:events.append(dict(time_s=t,kind='STOP',reason=hard));stop=hard
        if phase in ('LIFT','HAND_PREPARATION','TRANSFER') and not hard:
            nominal=path.sample(cursor)
            scale,lag=tracking_progress_scale(measured,last,nominal)
            cursor=min(path.duration_s,cursor+DT*scale)
            nominal=path.sample(cursor)
            index=min(int(np.searchsorted(ends,cursor,side='right')),len(ends)-1)
            new_phase=str(phases[index])
            if new_phase!=phase:events.append(dict(time_s=t,kind='PHASE',phase=new_phase,path_elapsed_s=cursor));phase=new_phase
            if cursor>=path.duration_s:
                phase='ENDPOINT_SETTLE';phase_started=t;settle_since=None
                events.append(dict(time_s=t,kind='ROUTE_ENDPOINT',tracking_error_rad=lag))
        elif phase=='ENDPOINT_SETTLE' and not hard:
            nominal=path.waypoints[-1].copy()
            lag=float(np.max(abs(measured[:7]-last[:7])))
            settle_since=(t if settle_since is None else settle_since) if lag<=.004 else None
            if t-phase_started>=.15 and settle_since is not None and t-settle_since>=.1:
                phase='REACQUIRE';phase_started=t;nominal[7:]=measured[7:];reacquire=nominal.copy();offset=0.
                fingers.rebase(nominal);mcc=GraceMCC(h,DT)
                events.append(dict(time_s=t,kind='REACQUIRE_BEGIN'))
        elif phase in ('REACQUIRE','CONTACT_HOLD') and not hard:
            nominal=reacquire.copy()
            if np.any(safe):stable_since=t if stable_since is None else stable_since
            else:
                stable_since=None
                if not relief_active:offset=min(.006,offset+.0015*DT)
            nominal[:7]+=endpoint_arm_offset(h,data,normal,offset)
            if phase=='REACQUIRE' and stable_since is not None and t-stable_since>=.15:
                phase='CONTACT_HOLD';events.append(dict(time_s=t,kind='RECONTACT_STABLE',offset_m=offset))
            elif phase=='REACQUIRE' and t-phase_started>8:
                stop='REACQUIRE_WINDOW_EXHAUSTED';hard=stop
        enabled=phase in ('REACQUIRE','CONTACT_HOLD') and not relief_active and not hard
        finger_nominal,diag=fingers.update(data,nominal,forces,normals,active,enabled=enabled)
        command=mcc.command(data,finger_nominal,forces,normals,active,2.) if enabled else finger_nominal.copy()
        if relief_active:
            command=relief.relieve_force(h,data,command,forces,normals)
            command[:7]=arm_force_release(h,data,last,forces,normals)
        command=np.clip(np.clip(command,last-.0024,last+.0024),planner.lower,planner.upper)
        if step%5==0 and not hard:
            reason=planner.command_collision_reason(measured,command)
            if reason in ('NON_TIP_CLEARANCE','NON_TIP_OBJECT_COLLISION','SELF_COLLISION','ARM_SELF_CLEARANCE','SHALLOW_ARM_SELF_COLLISION'):
                hard='COMMAND_GEOMETRY_'+reason;stop=hard;events.append(dict(time_s=t,kind='STOP',reason=hard,phase=phase))
        if hard:command=last.copy()
        values=dict(time_s=t,native_time_s=float(data.time),q_rad=measured.copy(),
            qvel_rad_s=np.r_[data.qvel[h.arm_dof_adrs],data.qvel[h.hand_dof_adrs]],qacc_warmstart=data.qacc_warmstart.copy(),
            command_rad=command.copy(),sensor_command_rad=last.copy(),palm_pose_world=pose(h,data),tip_positions_world_m=tip_positions(h,data),
            contact_positions_world_m=points.copy(),normals_world=normals.copy(),forces_n=forces.copy(),active=active.copy(),
            phase=observed_phase,command_phase=phase,route_elapsed_s=cursor,endpoint_offset_m=offset,
            arm_tracking_error_rad=float(np.max(abs(measured[:7]-last[:7]))),arm_self_force_n=self_force,
            coverage=coverage.area_fraction,covered_goals=int(coverage.visited_goals.sum()),covered_vertex_count=int(coverage.visited_vertices.sum()),
            new_vertex_count=int(np.count_nonzero(coverage.visited_vertices&~initial_mask)),goal_id=goal_vertex,
            target_region_safe_contact=near_goal,
            finger_search_active=np.asarray(diag['seeking']),finger_search_state=np.asarray(diag['state']),
            finger_search_bias_rad=np.asarray(diag['joint_bias_rad']),finger_search_enabled=enabled,
            finger_search_distance_m=np.asarray(diag['search_distance_m']),finger_search_delta_rad=np.asarray(diag['search_delta_rad']))
        for key,value in values.items():logs.setdefault(key,[]).append(value)
        if step%2500==0:
            print('TERMINAL_PROBE',json.dumps(dict(time_s=t,phase=phase,path_s=cursor,forces_n=forces.tolist(),new_vertices=values['new_vertex_count'])),flush=True)
            (directory/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
        if coverage.visited_vertices.all() and near_goal and goal_confirm is not None:
            stop='COMPOSITE_CONTINUATION_FULL_ORIGINAL_SURFACE';break
        if hard or step==int(duration_s/DT):break
        last=command.copy();set_control(h,data,command);mujoco.mj_step(h.model,data);mujoco.mj_forward(h.model,data)
    arrays={k:np.asarray(v) for k,v in logs.items()};arrays['covered_vertices']=coverage.visited_vertices.copy()
    arrays['required_goal_visited']=coverage.visited_goals.copy();arrays['source_covered_vertices']=initial_mask
    np.savez_compressed(directory/'trace.npz',**arrays)
    np.savez_compressed(directory/'final_state.npz',qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl,qacc_warmstart=data.qacc_warmstart,time=data.time)
    summary=dict(scope=manifest['scope'],stop_reason=stop,executed_s=t,wall_s=perf_counter()-started,
        native_initial_time_s=initial_time,native_final_time_s=float(data.time),planned_duration_s=path.duration_s,route_elapsed_s=cursor,
        peak_force_n=peak,peak_arm_self_force_n=peak_arm,first_goal_region_contact_s=first_goal_touch,goal_region_stable_confirmation_s=goal_confirm,
        new_covered_vertices=np.flatnonzero(coverage.visited_vertices&~initial_mask).tolist(),remaining_vertices=np.flatnonzero(~coverage.visited_vertices).tolist(),
        combined_covered_vertex_count=int(coverage.visited_vertices.sum()),combined_area_fraction=coverage.area_fraction,
        combined_visited_required_goals=int(coverage.visited_goals.sum()),fresh_full_traversal_success=False,
        trace_sha256=_hash(directory/'trace.npz'))
    (directory/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
    (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('TERMINAL_PROBE_RESULT',json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-run',type=Path,required=True);parser.add_argument('--route',type=Path,required=True)
    parser.add_argument('--duration',type=float,default=70.)
    args=parser.parse_args();run_probe(args.output,args.source_run,args.route,duration_s=args.duration)
