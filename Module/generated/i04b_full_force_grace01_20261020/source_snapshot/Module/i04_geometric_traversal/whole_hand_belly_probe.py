"""Native cooperative four-finger acquisition and curved-surface sliding.

This is a new local experiment, never an inherited whole-object result. The
unchanged plant is initialized once at a declared collision-free hover.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
import mujoco
import numpy as np
import trimesh

from Module.module_2_fingertip_mcc import FingertipMCC, MCCConfig
from .whole_hand_belly import WholeHandBellyPlanner, contact_clearance_handoff, cooperative_window_ready
from .natural_belly_traversal import set_kinematics, unit
from .belly_contact import BellyContactConfig, belly_contact_measurement, belly_ik_terms
from .belly_bounded_step import bounded_belly_step
from .contracts import ExperimentConfig
from .full_surface import build_scene
from .physics import set_joint_state, set_control, joint_state, contact_measurement
from .sliding_safety import arm_self_contacts
from .oracle import MeasuredCoverage


def snapshot_sources(out):
    root = Path(__file__).resolve().parents[2]; sources = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(root/'Module'):
                relative = path.relative_to(root).as_posix(); raw = path.read_bytes()
                sources[relative] = sha256(raw).hexdigest()
                dest = out/'source_snapshot'/relative; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(raw)
    return sources


def run(output, geometry, index=4, seconds=30., stroke_m=.06):
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
    h, graph, _, quality = build_scene(out); planner = WholeHandBellyPlanner(h, graph)
    geometry = Path(geometry).resolve()
    with np.load(geometry) as z:seed = z['q'][index].copy()
    q, initial_plan, points, normals = planner.fit(seed, clearance=.006)
    if not initial_plan['accepted']:raise RuntimeError('NO_COOPERATIVE_HOVER:'+json.dumps(initial_plan))
    data = mujoco.MjData(h.model); scratch = mujoco.MjData(h.model)
    set_joint_state(h, data, q); set_control(h, data, q)
    if contact_measurement(h, data, .2)[0].max()>.05:raise RuntimeError('INITIAL_HOVER_ALREADY_LOADED')
    (out/'model.xml').write_text(h.xml, encoding='utf-8')
    np.savez_compressed(out/'initial.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl,
        qacc_warmstart=data.qacc_warmstart, time=data.time)
    cfg = BellyContactConfig(); coverage = MeasuredCoverage(graph, ExperimentConfig())
    manifest = dict(protocol='i04-whole-hand-belly.development.v1.6', scope='LOCAL_COOPERATIVE_SLIDE', cell='B-MCC',
        dp='DEFERRED_BY_USER', source_hashes=snapshot_sources(out), geometry_quality=quality,
        belly_contact=asdict(cfg), posture_lower_rad=planner.lower.tolist(), posture_upper_rad=planner.upper.tolist(),
        initial_plan=initial_plan, initial_state='One declared cooperative-hand 6 mm hover; zero inherited coverage',
        geometry_source=str(geometry), geometry_source_sha256=sha256(geometry.read_bytes()).hexdigest(), geometry_index=index,
        coverage_inherited=False, resume=None, coverage_radius_m=.025, required_vertices=8192, required_goals=388,
        original_collision_geometry=True, command_reference_guard_every_native_tick=True,
        command_geometry_guard_every_native_tick=True, dt_s=.002, proactive_release_n=2.5,
        release_resume_n=1.2, force_relief_n=8., hard_force_n=12., hard_force_duration_s=.020,
        controlled_fingers=[0,1,2,3], desired_force_per_finger_n=1., requested_duration_s=seconds,
        stroke_m=stroke_m, simultaneous_contacts_to_start=4, simultaneous_contacts_to_advance=3,
        cooperative_window_s=.3, window_three_contact_fraction=.9, window_each_finger_fraction=.6,
        camera_focus_finger=1, leader_finger_field_is_visualization_only=True, continuous_each_finger_contact_handoff=True,
        mcc_damping=70., mcc_stiffness=20., mcc_normal_velocity_cap_m_s=.00025,
        tangential_target_weight=.35, nonpad_soft_clearance_m=.006, shared_nominal_wrist_route=True,
        qualified_belly_normal_first=True,tangential_seam_escape_max_m=.006,tangential_seam_escape_speed_m_s=.001,
        velocity_release_lookahead_s=.04, release_normal_reference_increment_cap_m=.0006)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    travel = np.linspace(0., stroke_m, max(2, int(np.ceil(stroke_m/.001))+1))
    direction = np.array([0., 1., 0.]); mean_normal = unit(normals.mean(axis=0))
    direction = unit(direction-mean_normal*(mean_normal@direction))
    path_points = []; path_normals = []
    for finger in range(4):
        raw = points[finger]+travel[:,None]*direction
        projected, _, _ = trimesh.proximity.closest_point(graph.mesh, raw-h.object_position_m)
        path_points.append(projected+h.object_position_m)
        path_normals.append(np.array([planner.smooth_normal(p) for p in projected]))
    path_points = np.asarray(path_points); path_normals = np.asarray(path_normals)
    nominal_q=[];planned=q.copy();nominal_checks=[]
    for index in range(len(travel)):
        planned,check=planner.fit_targets(planned,path_points[:,index],path_normals[:,index])
        nominal_q.append(planned);nominal_checks.append(check)
    nominal_q=np.asarray(nominal_q)
    np.savez_compressed(out/'planned_patch.npz', travel=travel, points=path_points, normals=path_normals, initial_q=q, nominal_q=nominal_q)
    (out/'nominal_geometry_checks.json').write_text(json.dumps(nominal_checks,indent=2))
    controllers = [FingertipMCC(MCCConfig(virtual_mass=.08, damping=70., stiffness=20., dt_s=.002,
        max_offset_m=.004, max_velocity_m_s=.00025, max_acceleration_m_s2=.02)) for _ in range(4)]
    last = q.copy(); offsets = np.zeros(4); normal_filter = normals.copy(); clearance = np.full(4,.006)
    logs = {}; events = []; chunks = []; states = 0; peak = 0.; high_steps = 0
    simultaneous_hold = 0.; progress = 0.; slide_start = None; release = np.zeros(4,bool)
    contact_window=deque(maxlen=150)
    tangent_bias=np.zeros((4,3))
    start = perf_counter(); last_print = start; stop = 'UNFINISHED'; geometry_rejections = 0

    def save_chunk():
        nonlocal logs
        if not logs:return
        name = f'trace_{len(chunks):04d}.npz'
        np.savez_compressed(out/name, **{k:np.asarray(v) for k,v in logs.items()})
        chunks.append(dict(file=name, states=len(logs['time_s']), start_s=float(logs['time_s'][0]),
            end_s=float(logs['time_s'][-1]), sha256=sha256((out/name).read_bytes()).hexdigest()))
        logs = {}
        np.savez_compressed(out/'checkpoint.npz', qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl,
            qacc_warmstart=data.qacc_warmstart, time=data.time, visited_vertices=coverage.visited_vertices,
            visited_goals=coverage.visited_goals)
        (out/'trace_chunks.json').write_text(json.dumps(chunks, indent=2))
        (out/'events.json').write_text(json.dumps(events, indent=2))

    try:
        while data.time<seconds:
            f, p, n, active, unsafe, penetration = contact_measurement(h, data, .2)
            bf, bp, bn, ba, bd = belly_contact_measurement(h, data, .2, cfg)
            rejected = np.asarray(bd['rejected_contact_force_n']); arm = arm_self_contacts(h, data)
            safe = ba & (f<8.) & (rejected<.2)
            if unsafe or arm or f.max()>=8.:safe[:]=False
            coverage.update(bp-h.object_position_m, bn, safe)
            peak = max(peak, float(f.max())); high_steps = high_steps+1 if f.max()>12. else 0
            simultaneous_hold = simultaneous_hold+.002 if safe.all() else 0.
            contact_window.append(safe.copy());cooperative_ready=cooperative_window_ready(contact_window)
            if slide_start is None and cooperative_ready:
                slide_start = float(data.time); events.append(dict(kind='FOUR_BELLY_SLIDE_BEGIN', time_s=data.time))
            phase = 'COOPERATIVE_APPROACH' if slide_start is None else 'COOPERATIVE_SLIDE'
            release |= f>2.5; release &= f>=1.2
            if release.any():phase = 'FORCE_RELEASE'
            target_points = np.array([[np.interp(progress,travel,path_points[i,:,j]) for j in range(3)] for i in range(4)])
            for finger in range(4):
                if ba[finger] and rejected[finger]>=.2:
                    away=np.zeros(3)
                    for contact in bd['records']:
                        if contact['finger']==finger and not contact['geometry_qualified']:
                            outward=np.asarray(contact['object_outward_normal_world'])
                            away+=contact['normal_force_n']*(outward-bn[finger]*(outward@bn[finger]))
                    if np.linalg.norm(away)>1e-8:
                        tangent_bias[finger]+=.001*.002*unit(away)
                        norm=np.linalg.norm(tangent_bias[finger])
                        if norm>.006:tangent_bias[finger]*=.006/norm
            target_points+=tangent_bias
            planned_normals = np.array([unit([np.interp(progress,travel,path_normals[i,:,j]) for j in range(3)]) for i in range(4)])
            set_kinematics(h, scratch, last)
            errors = []; rows = []; details = []
            for finger in range(4):
                if ba[finger] and bn[finger]@planned_normals[finger]>.7:normal_target=bn[finger]
                else:normal_target = n[finger] if active[finger] and n[finger]@planned_normals[finger]>.7 else planned_normals[finger]
                angle = np.arccos(np.clip(normal_filter[finger]@normal_target,-1.,1.))
                blend = min(.06,.35*.002/max(angle,1e-12))
                normal_filter[finger] = unit((1-blend)*normal_filter[finger]+blend*normal_target)
                normal = normal_filter[finger]
                # Contacting fingers stop descending independently; late fingers
                # keep approaching, without lowering the entire hand onto early fingers.
                if safe[finger] and clearance[finger]>0.:
                    clearance[finger], offsets[finger] = contact_clearance_handoff(clearance[finger], offsets[finger])
                    controllers[finger].reset(offset_m=offsets[finger])
                else:clearance[finger] = max(0.,clearance[finger]-.003*.002)
                prior = offsets[finger]
                if clearance[finger]<.0005:
                    desired=(.75 if ba[finger] else 0.) if rejected[finger]>=.2 else 1.
                    offset = controllers[finger].step(target_points[finger],-normal,desired,float(f[finger])).offset_m
                    offsets[finger] = offset
                if release[finger]:
                    set_kinematics(h, planner.data, joint_state(h,data))
                    _, actual_j, _ = belly_ik_terms(h.model,planner.data,int(h.tip_geom_ids[finger]),
                        target_points[finger],normal,dof_indices=planner.order)
                    velocity = float(normal@actual_j[:3]@data.qvel[planner.order])
                    outward = min(.0006,.00004+max(0.,-velocity)*.04)
                    offsets[finger] = max(-.004, prior-outward)
                    controllers[finger].reset(offset_m=offsets[finger])
                e, j, detail = belly_ik_terms(h.model,scratch,int(h.tip_geom_ids[finger]),target_points[finger],normal,
                    clearance_m=clearance[finger]-offsets[finger],orientation_weight_m=.04,dof_indices=planner.order)
                metric = np.outer(normal,normal)+.35*(np.eye(3)-np.outer(normal,normal))
                errors.append(np.r_[metric@e[:3],e[3:]])
                rows.append(np.vstack((metric@j[:3],j[3:]))); details.append(detail)
            nominal=np.array([np.interp(progress,travel,nominal_q[:,j]) for j in range(23)])
            if slide_start is None:
                fraction=min(1.,float(data.time)/2.)
                nominal=(1-fraction)*q+fraction*nominal
            pe, pj = planner.posture_terms(last,nominal,arm_weight=.03,finger_weight=.0025)
            errors.append(pe); rows.append(pj)
            gj, ge = planner.arm_buffer_terms(last,np.arange(23))
            if len(ge):errors.append(ge); rows.append(gj)
            oe, oj = planner.object_buffer_terms(last)
            if len(oe):errors.append(oe); rows.append(oj)
            delta, diagnostic = bounded_belly_step(np.vstack(rows),np.concatenate(errors),last,planner.lower,planner.upper,
                gain=.55,maximum_joint_step=np.r_[np.full(7,.0004),np.full(16,.0008)])
            if not diagnostic['feasible']:raise RuntimeError('NO_COOPERATIVE_STEP:'+diagnostic['reason'])
            proposed = last+delta
            def reason(command):
                return planner.safety.command_collision_reason(joint_state(h,data),command) or planner.safety.command_collision_reason(last,command)
            initial_reason = reason(proposed); command = proposed
            if initial_reason:
                geometry_rejections += 1; command = None
                for scale in (.5,.25,.125,0.):
                    trial = last+scale*delta
                    if reason(trial) is None:command = trial;break
                if command is None:
                    command = planner.clearance_exit(joint_state(h,data),last)
                    if command is None or reason(command) is not None:raise RuntimeError('NO_GEOMETRY_SAFE_COMMAND:'+initial_reason)
                    phase = 'CLEARANCE_EXIT'
            assert np.all(command>=planner.lower) and np.all(command<=planner.upper)
            assert np.max(abs(command-last))<=.0024+1e-10
            values = dict(time_s=data.time,q_rad=joint_state(h,data),qvel_rad_s=data.qvel.copy(),qacc_warmstart=data.qacc_warmstart.copy(),
                sensor_command_rad=last.copy(),command_rad=command.copy(),forces_n=f,contact_positions_world_m=p,normals_world=n,active=active,
                belly_forces_n=bf,belly_contact_positions_world_m=bp,belly_normals_world=bn,belly_active=ba,rejected_contact_force_n=rejected,
                safe_belly=safe,phase=phase,leader_finger=1,mcc_offset_m=offsets.copy(),coverage=coverage.area_fraction,
                visited_vertex_count=int(coverage.visited_vertices.sum()),unsafe_count=len(unsafe),
                arm_self_force_n=max([c['force_n'] for c in arm],default=0.),actual_penetration_m=penetration,
                stroke_distance_m=progress,controlled_fingers=np.ones(4,bool),finger_search_active=~safe,
                simultaneous_safe_contacts=int(safe.sum()),reference_geometry_backtrack=initial_reason or '',
                target_points_world=target_points,target_normals_world=normal_filter.copy(),
                target_clearance_m=clearance.copy(),belly_alignment_deg=np.array([d['alignment_angle_deg'] for d in details]))
            values['tangential_seam_bias_m']=tangent_bias.copy()
            for name,value in values.items():logs.setdefault(name,[]).append(value)
            states += 1
            if unsafe or arm:raise RuntimeError('UNSAFE_NATIVE_CONTACT')
            if high_steps>=10:raise RuntimeError('SUSTAINED_DANGEROUS_FORCE')
            last = command.copy();set_control(h,data,last);mujoco.mj_step(h.model,data);mujoco.mj_forward(h.model,data)
            if slide_start is not None and safe.sum()>=3 and not release.any() and rejected.max()<.2:
                progress = min(stroke_m,progress+.004*.002)
            if progress>=stroke_m and cooperative_ready:
                stop = 'COOPERATIVE_LOCAL_STROKE_COMPLETED';break
            if states%5000==0:save_chunk()
            if perf_counter()-last_print>12.:
                status=dict(time_s=float(data.time),progress_m=progress,phase=phase,forces_n=f.tolist(),safe=safe.tolist(),peak_n=peak)
                (out/'progress.json').write_text(json.dumps(status,indent=2));print('COOPERATIVE',json.dumps(status),flush=True);last_print=perf_counter()
            if states%500==0 and (out/'development_stop.json').exists():raise RuntimeError('DEVELOPMENT_STOP_REQUESTED')
        else:stop='COOPERATIVE_ACQUISITION_TIMEOUT' if slide_start is None else 'COOPERATIVE_STROKE_TIMEOUT'
    except RuntimeError as error:stop=str(error)
    finally:
        save_chunk()
        np.savez_compressed(out/'final_state.npz',qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl,qacc_warmstart=data.qacc_warmstart,
            time=data.time,visited_vertices=coverage.visited_vertices,visited_goals=coverage.visited_goals)
        result=dict(stop_reason=stop,executed_s=float(data.time),full_surface_complete=False,
            coverage_fraction=coverage.area_fraction,visited_vertices=int(coverage.visited_vertices.sum()),
            total_vertices=8192,visited_goals=int(coverage.visited_goals.sum()),total_goals=388,
            native_states=states,routes=1,necessary_lifts=0,peak_force_n=peak,coverage_inherited=False,fresh_single_version=True,
            resume=None,trace_chunks=chunks,cooperative_slide_start_s=slide_start,stroke_completed_m=progress,
            geometry_backtracks=geometry_rejections,cooperative_local_complete=stop=='COOPERATIVE_LOCAL_STROKE_COMPLETED')
        (out/'summary.json').write_text(json.dumps(result,indent=2));print('RESULT',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--geometry',required=True)
    parser.add_argument('--index',type=int,default=4);parser.add_argument('--seconds',type=float,default=30.)
    parser.add_argument('--stroke',type=float,default=.06);args=parser.parse_args()
    run(args.output,args.geometry,args.index,args.seconds,args.stroke)
