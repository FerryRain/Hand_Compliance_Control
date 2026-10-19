"""Native development experiment: face the palmar pad, reacquire, then slide.

Original mustard, collision geometry and force guards. This local experiment
does not claim whole-bottle completion and does not reuse historical coverage.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
import trimesh

from Module.module_2_fingertip_mcc import FingertipMCC, MCCConfig
from .belly_contact import BellyContactConfig, belly_contact_measurement, belly_ik_terms
from .belly_surface_planner import BellySurfacePlanner
from .belly_bounded_step import bounded_belly_step
from .contracts import ExperimentConfig, WristTarget
from .full_surface import build_scene, acquire
from .physics import joint_state, set_joint_state, set_control, pose, contact_measurement, tip_positions
from .sliding_safety import arm_self_contacts
from .oracle import MeasuredCoverage


def common_scale(last, desired, lower, upper, maximum=.0024):
    delta=desired-last; alpha=1.
    for k,value in enumerate(delta):
        if value>0: alpha=min(alpha,maximum/value,(upper[k]-last[k])/value)
        elif value<0: alpha=min(alpha,-maximum/value,(lower[k]-last[k])/value)
    if alpha<0: raise RuntimeError('REFERENCE_OUTSIDE_JOINT_BOX')
    alpha=max(0.,min(1.,alpha))
    if 0<alpha<1: alpha=float(np.nextafter(alpha,0.))
    return last+alpha*delta


def run(directory, slide_s=12.):
    out=Path(directory).resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    root=Path(__file__).resolve().parents[2]
    sources={}
    for mod in list(sys.modules.values()):
        filename=getattr(mod,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.suffix=='.py' and path.is_relative_to(root/'Module'):
                rel=str(path.relative_to(root)).replace('\\','/')
                raw=path.read_bytes(); sources[rel]=sha256(raw).hexdigest()
                target=out/'source_snapshot'/rel; target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    cfg=ExperimentConfig(); belly_cfg=BellyContactConfig()
    h,graph,q,quality=build_scene(out)
    data,initial=acquire(h,graph,q)
    bp=BellySurfacePlanner(h,graph)
    (out/'model.xml').write_text(h.xml,encoding='utf-8')
    np.savez_compressed(out/'initial.npz',qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl,
        qacc_warmstart=data.qacc_warmstart,time=data.time)
    manifest=dict(protocol='i04-belly-surface-local.development.v1.1',cell='B-MCC',dp='DEFERRED_BY_USER',
        source_hashes=sources,initial_state=initial,geometry_quality=quality,belly_contact=asdict(belly_cfg),
        original_raw_force_guards_retained=True,additional_local_stop_n=8.,full_surface_claim=False,coverage_inherited=False,
        command_geometry_guard_every_native_tick=True,
        local_slide_requested_s=slide_s,normal_mcc=dict(desired_force_n=1.,maximum_offset_m=.0016,
            maximum_velocity_m_s=.004,virtual_mass=.08,damping=30.,stiffness=40.))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    coverage=MeasuredCoverage(graph,cfg)
    logs={};events=[]; t0=data.time;last=np.r_[data.ctrl[h.arm_actuator_ids],data.ctrl[h.hand_actuator_ids]]
    high_steps=0;peak=0.;stop='UNFINISHED';current_phase='INITIAL';offsets=np.zeros(4)
    def step(command,phase,advance=0.):
        nonlocal last,high_steps,peak,current_phase
        current_phase=phase
        f,p,n,a,unsafe,pen=contact_measurement(h,data,.2)
        bf,bpnt,bn,ba,bd=belly_contact_measurement(h,data,.2,belly_cfg)
        arm=arm_self_contacts(h,data)
        peak=max(peak,float(f.max())); high_steps=high_steps+1 if f.max()>12 else 0
        safe=ba & (f<8) & (np.asarray(bd['rejected_contact_force_n'])<.2)
        if unsafe or arm or f.max()>=8:safe[:]=False
        coverage.update(bpnt-h.object_position_m,bn,safe)
        command=common_scale(last,command,bp.safety.lower,bp.safety.upper)
        values=dict(time_s=data.time-t0,q_rad=joint_state(h,data),
            qvel_rad_s=data.qvel.copy(),qacc_warmstart=data.qacc_warmstart.copy(),
            sensor_command_rad=last.copy(),command_rad=command.copy(),
            palm_pose_world=pose(h,data),tip_positions_world_m=tip_positions(h,data),
            contact_positions_world_m=p,forces_n=f,normals_world=n,active=a,
            belly_forces_n=bf,belly_contact_positions_world_m=bpnt,belly_normals_world=bn,belly_active=ba,
            rejected_contact_force_n=np.asarray(bd['rejected_contact_force_n']),
            mcc_offset_m=offsets.copy(),
            phase=phase,stroke_distance_m=advance,coverage=coverage.area_fraction,
            pad_frames_world=data.geom_xmat[h.tip_geom_ids].copy(),
            finger_search_active=(~ba & np.array([1,1,1,0],bool) & (phase in ('BELLY_APPROACH','BELLY_SLIDE'))),
            finger_search_state=np.where(np.arange(4)==3,'NOT_CONTROLLED',
                np.where(phase in ('BELLY_APPROACH','BELLY_SLIDE'),np.where(ba,'BELLY_HOLD','SEEK'),'PLANNED_POSTURE')),
            unsafe_count=len(unsafe),arm_self_force_n=max([c['force_n'] for c in arm],default=0.))
        for key,value in values.items():logs.setdefault(key,[]).append(value)
        if unsafe or arm:raise RuntimeError('UNSAFE_NATIVE_CONTACT')
        if high_steps>=10:raise RuntimeError('SUSTAINED_DANGEROUS_FORCE')
        # Local development additionally stops at relief threshold. All force,
        # including rejected side/back contact, remains visible to this guard.
        if f.max()>=8:raise RuntimeError('FORCE_RELIEF_REQUIRED_LOCAL_STOP')
        reason=bp.safety.command_collision_reason(joint_state(h,data),command)
        if reason:
            events.append(dict(time_s=data.time-t0,kind='COMMAND_GEOMETRY_REJECTION',reason=reason,
                mcc_offset_m=offsets.tolist(),command_clearance_depth=bp.safety.clearance_state()))
            raise RuntimeError('COMMAND_GEOMETRY:'+reason)
        last=command.copy();set_control(h,data,last)
        mujoco.mj_step(h.model,data);mujoco.mj_forward(h.model,data)
        return f,p,n,a,bf,ba
    def follow(path,phase):
        events.append(dict(time_s=data.time-t0,kind=phase,duration_s=path.duration_s))
        begin=data.time
        while data.time-begin<path.duration_s+.25:
            step(path.sample(data.time-begin),phase)
        print(phase,'finished',round(data.time-t0,3),flush=True)
    try:
        f,points,normals,active,_,_=contact_measurement(h,data,.2)
        fingers=[0,1,2]
        targets={i:(points[i].copy(),normals[i].copy()) for i in fingers}
        # Joint optimization of all three fingers avoids rotating one finger's
        # wrist through its neighbours. Test the smallest listed retreat first.
        choice=None
        for distance in (.004,.008,.012,.020):
            start=joint_state(h,data); wrist=pose(h,data)
            direction=np.mean(normals[active],axis=0);direction/=np.linalg.norm(direction)
            wrist[:3]+=distance*direction
            retreat,ep,er,_=bp.safety.solve(start,WristTarget(wrist))
            path,ed=bp.path(start,retreat)
            aligned,diag=bp.solve(retreat,targets,clearance=distance)
            align_path,ad=bp.path(retreat,aligned)
            events.append(dict(kind='INITIAL_FACE_PLAN',retreat_m=distance,ik=diag,retreat_edge=ed,align_edge=ad))
            if path and align_path and diag['accepted']:
                choice=distance,path;break
        if choice is None:raise RuntimeError('NO_SAFE_BELLY_ORIENTATION_PLAN')
        distance,path=choice
        follow(path,'UNLOAD_WRONG_FACE')
        actual=joint_state(h,data)
        aligned,diag=bp.solve(actual,targets,clearance=distance)
        align_path,ad=bp.path(actual,aligned)
        if not diag['accepted'] or align_path is None:raise RuntimeError('MEASURED_ALIGNMENT_REPLAN_FAILED')
        follow(align_path,'BELLY_ALIGN')
        # Online support-position/normal control lets joints move throughout
        # approach and sliding. MCC offset is integrated from native forces.
        controllers=[FingertipMCC(MCCConfig(virtual_mass=.08,damping=30.,stiffness=40.,
            dt_s=.002,max_offset_m=.0016,max_velocity_m_s=.004,max_acceleration_m_s2=1.)) for _ in fingers]
        scratch=mujoco.MjData(h.model)
        columns=np.arange(19);dofs=bp.order[columns]
        begin=data.time; stable=0.;slide_distance=0.; slide_begin=None
        travel=np.arange(0.,.08001,.001)
        direction=np.array([0.,0.,1.]);mean_n=np.mean(normals[fingers],axis=0)
        direction-=mean_n*(direction@mean_n)/(mean_n@mean_n);direction/=np.linalg.norm(direction)
        surface=[];surface_normals=[]
        for i in fingers:
            raw=points[i]+travel[:,None]*direction
            projected,_,faces=trimesh.proximity.closest_point(graph.mesh,raw-h.object_position_m)
            ns=[]
            for point in projected:
                dist,ind=graph._tree.query(point,k=12)
                normal=np.sum(graph.normals[ind]/np.maximum(dist[:,None],.002),axis=0)
                ns.append(normal/np.linalg.norm(normal))
            surface.append(projected+h.object_position_m);surface_normals.append(np.asarray(ns))
        while data.time-begin<25.+slide_s:
            elapsed=data.time-begin
            f,p,n,a,_,_=contact_measurement(h,data,.2)
            bf,bpnt,bn,ba,bd=belly_contact_measurement(h,data,.2,belly_cfg)
            # Brief loss pauses progress and actively seeks at the current
            # surface point; it is not an experiment failure.
            stable=stable+.002 if np.count_nonzero(ba[fingers])>=2 else 0.
            if slide_begin is None and stable>=.15:
                slide_begin=data.time;events.append(dict(time_s=data.time-t0,kind='BELLY_SLIDE_BEGIN'))
            phase='BELLY_APPROACH' if slide_begin is None else 'BELLY_SLIDE'
            if slide_begin is not None:
                wrong_faces=np.asarray(bd['rejected_contact_force_n'])[fingers]>=.2
                if np.count_nonzero(ba[fingers])>=2 and not np.any(wrong_faces):
                    slide_distance+=.004*.002
                if data.time-slide_begin>=slide_s:break
            clearance=max(0.,distance-.003*elapsed)
            set_joint_state(h,scratch,last)
            errors=[];jacobians=[]
            for k,i in enumerate(fingers):
                point=np.array([np.interp(slide_distance,travel,surface[k][:,j]) for j in range(3)])
                normal=np.array([np.interp(slide_distance,travel,surface_normals[k][:,j]) for j in range(3)])
                normal/=np.linalg.norm(normal)
                # Start compliance after approaching the surface. Free fingers
                # receive the same bounded inward search, not a fixed posture.
                offset=0.
                if clearance<.0005:
                    wrong_face=bool(bd['rejected_contact_force_n'][i]>=.2)
                    offset=controllers[k].step(point,-normal,0. if wrong_face else 1.,float(f[i])).offset_m
                offsets[i]=offset
                residual,jac,diag=belly_ik_terms(h.model,scratch,int(h.tip_geom_ids[i]),point,normal,
                    clearance_m=clearance-offset,orientation_weight_m=.025,dof_indices=dofs)
                errors.append(residual);jacobians.append(jac)
            J=np.vstack(jacobians);error=np.concatenate(errors)
            delta,step_diag=bounded_belly_step(J,error,last[columns],
                bp.safety.lower[columns],bp.safety.upper[columns])
            if not step_diag['feasible']:
                events.append(dict(time_s=data.time-t0,kind='BOUNDED_STEP_REJECTION',diagnostic=step_diag))
                np.savez_compressed(out/'rejected_step.npz',jacobian=J,error=error,
                    previous=last[columns],lower=bp.safety.lower[columns],upper=bp.safety.upper[columns])
                raise RuntimeError('BOUNDED_BELLY_STEP:'+step_diag['reason'])
            command=last.copy();command[columns]+=delta
            step(command,phase,slide_distance)
            if len(logs['time_s'])%2500==0:
                print('BELLY',round(data.time-t0,2),'s',phase,'distance',round(slide_distance,4),
                      'force',f.tolist(),'belly',ba.tolist(),flush=True)
        stop='LOCAL_SLIDE_FINISHED' if slide_begin is not None else 'BELLY_ACQUISITION_TIMEOUT'
    except RuntimeError as error:
        stop=str(error)
    finally:
        arrays={key:np.asarray(value) for key,value in logs.items()}
        np.savez_compressed(out/'trace.npz',**arrays)
        np.savez_compressed(out/'final_state.npz',qpos=data.qpos,qvel=data.qvel,ctrl=data.ctrl,
            qacc_warmstart=data.qacc_warmstart,time=data.time)
        (out/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
        result=dict(stop_reason=stop,executed_s=data.time-t0,peak_force_n=peak,
            full_surface_complete=False,belly_coverage_fraction=coverage.area_fraction,
            trace_sha256=sha256((out/'trace.npz').read_bytes()).hexdigest())
        if arrays:
            slide=arrays['phase']=='BELLY_SLIDE'
            result.update(native_states=len(arrays['time_s']),
                slide_distance_m=float(arrays['stroke_distance_m'].max()),
                belly_contact_fraction_by_finger=np.mean(arrays['belly_active'],axis=0).tolist(),
                slide_belly_contact_fraction_by_finger=(np.mean(arrays['belly_active'][slide],axis=0).tolist() if slide.any() else None),
                actual_finger_joint_range_deg=np.degrees(np.ptp(arrays['q_rad'][:,7:],axis=0)).tolist())
        (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('RESULT',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--slide-seconds',type=float,default=12.)
    a=p.parse_args();run(a.output,a.slide_seconds)
