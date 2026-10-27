"""Refine a full palm reference without hiding infeasible hand waypoints.

Every source palm pose and timestamp is preserved. The output is eligible
only after every serialized knot and interpolated frame passes geometry,
kinematic health, local motion reserve, and reference velocity checks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from analyze_grasp_clearance import smooth_keyframe_reference
from audit_healthy_grasp_plans import save
from healthy_grasp_planner import HealthyGraspPlanner,HealthConfig,DEFAULT_BANK,state_report
from plan_waypoint_viewer import PlanReference


class TemporalHealthyPlanner(HealthyGraspPlanner):
    temporal_limits=None

    def state(self,q,pose,capacity=False):
        result=super().state(q,pose,capacity)
        result['q']=np.asarray(q,float)
        return result

    def constraints(self,state):
        base=super().constraints(state)
        if self.temporal_limits is None:return base
        lo,hi=self.temporal_limits
        return np.r_[base,(state['q']-lo)/.2,(hi-state['q'])/.2]

    def advance(self,pose,previous,max_step):
        """Follow the moving palm with a small, bounded normal correction.

        Start at the previous feasible hand state. Do not independently
        maximize posture quality at every sample and interpolate between
        unrelated solutions. Every saved control sample is checked.
        """
        self.temporal_limits=(previous-max_step,previous+max_step)
        lo=np.maximum(self.lower,previous-max_step)
        hi=np.minimum(self.upper,previous+max_step)
        q=np.clip(previous,lo,hi)
        rotation=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
        for _ in range(5):
            s=self.state(q,pose)
            gap=s['geometry']['distance']
            if abs(gap-self.config.target_gap_m).max()<.00002 and self.valid(s):break
            normal=self.oracle.normals_at_world(s['geometry']['object_points']) @ rotation
            delta=np.zeros(16)
            for finger in range(4):
                block=slice(4*finger,4*finger+4)
                row=normal[finger] @ s['jacobians'][finger]
                delta[block]=(self.config.target_gap_m-gap[finger])*row/(row@row+1e-10)
            candidate=np.clip(q+delta,lo,hi)
            if abs(candidate-q).max()<1e-7:break
            q=candidate
        q=q.astype(np.float32).astype(float)
        state=self.state(q,pose,capacity=True)
        if not self.valid(state,include_capacity=True):
            # A difficult sample may need a redundant-joint adjustment.
            # The same hard bounds still apply; failed recovery is recorded.
            q,state=self.solve(pose,q,previous,multistart=1)
        else:
            state.update(valid=True,geometric_health_valid=True,
                objective=self.quality(state,q,previous))
        return q,state


def certify_dense(planner,reference,frames,q,max_velocity,output):
    # No valid-knot filtering: a bad knot may not disappear from acceptance.
    dense=smooth_keyframe_reference(frames,q,np.ones(len(q),bool),len(reference.poses))
    dense=dense.astype(np.float32).astype(float)
    rows=[];planner.temporal_limits=None
    for index,(value,pose) in enumerate(zip(dense,reference.poses)):
        state=planner.state(value,pose,capacity=True)
        constraint=planner.constraints(state)
        cap=state['capacity_m']
        severity=float(np.sum(np.minimum(constraint,0)**2)
            +np.sum((np.maximum(planner.config.min_normal_retreat_m-cap[:,0],0)/.001)**2)
            +np.sum((np.maximum(planner.config.min_tangent_motion_m-cap[:,1:],0)/.001)**2))
        severity+=float(np.count_nonzero(state['geometry'].get('cut_cap',[])))
        rows.append(dict(valid=planner.valid(state,include_capacity=True),severity=severity,
            pad_gap=state['geometry']['distance'],non_tip=state['geometry']['non_tip_distance'],
            self_penetration=state['geometry']['self_penetration'],health=state['health'],
            joint_margin=state['joint_margin_rad'],capacity=cap,
            origin_spread=state['geometry'].get('distance_origin_spread_m',np.zeros(4))))
    velocity=np.diff(dense,axis=0)/reference.dt
    valid=np.array([r['valid'] for r in rows],bool)
    velocity_fail=np.r_[False,(abs(velocity)>max_velocity+1e-3).any(-1)]
    valid&=~velocity_fail
    severity=np.array([r['severity'] for r in rows])
    severity+=np.r_[0.,np.maximum(abs(velocity).max(-1)-max_velocity,0)**2]
    np.savez_compressed(output,valid=valid,severity=severity,q=dense,
        pad_gap=np.stack([r['pad_gap'] for r in rows]),
        non_tip_distance=np.stack([r['non_tip'] for r in rows]),
        cross_finger_penetration=np.array([r['self_penetration'] for r in rows]),
        health=np.stack([r['health'] for r in rows]),
        joint_margin=np.stack([r['joint_margin'] for r in rows]),
        distance_origin_spread_m=np.stack([r['origin_spread'] for r in rows]),
        capacity_m=np.stack([r['capacity'] for r in rows]))
    return dense,valid,severity,dict(checked_frames=len(dense),passed=bool(valid.all()),
        valid_fraction=float(valid.mean()),first_invalid_frame=int(np.flatnonzero(~valid)[0]) if not valid.all() else None,
        max_reference_joint_velocity_rad_s=float(abs(velocity).max()),
        min_pad_gap_m=float(min(r['pad_gap'].min() for r in rows)),
        max_pad_gap_m=float(max(r['pad_gap'].max() for r in rows)),
        distance_query_origin_disagreement_frames=int(sum(r['origin_spread'].max()>2e-6 for r in rows)),
        min_normal_retreat_m=float(min(r['capacity'][:,0].min() for r in rows)),
        min_tangent_motion_m=float(min(r['capacity'][:,1:].min() for r in rows)))


def export_reference(source,output,reference,frames,q,states,dense,admissible,calibration):
    with h5py.File(source) as original,h5py.File(output,'w') as h:
        for key in original:
            if key.startswith(('grasp_keyframe_','finger_')) or key=='non_tip_geom_name':continue
            original.copy(key,h)
        for key,value in original.attrs.items():h.attrs[key]=value
        def write(key,value):
            if key in h:del h[key]
            h.create_dataset(key,data=value)
        write('q_hand',dense[:,None,:].astype(np.float32))
        write('qvel',np.r_[np.zeros((1,16)),np.diff(dense,axis=0)/reference.dt][:,None,:].astype(np.float32))
        write('grasp_keyframe_frame_index',np.asarray(frames)[:,None])
        write('grasp_keyframe_q',np.asarray(q,np.float32)[:,None,:])
        for key,values in dict(
            valid=[s['valid'] for s in states],true_tip_geometry_valid=[s['geometric_health_valid'] for s in states],
            contact_point_object=[s['geometry']['object_points'] for s in states],
            normal_object=[s['normal_object'] for s in states],
            signed_distance=[s['geometry']['distance'] for s in states],
            pad_normal_error=[np.arccos(np.clip(-np.sum(
                s['geometry']['pad_surface_normals']*s['normal_object'],axis=-1),-1.,1.)) for s in states],
            isotropic_reachability=[s['health'][0] for s in states],
            manipulability=[s['health'][1] for s in states],
            sigma_min=[s['health'][2] for s in states],
            non_tip_distance=[s['geometry']['non_tip_distance'] for s in states],
            cross_finger_penetration=[s['geometry']['self_penetration'] for s in states],
            local_motion_capacity_m=[s['capacity_m'] for s in states],
            tip_top_fraction=[s['geometry']['top_fraction'] for s in states]).items():
            write('grasp_keyframe_'+key,np.asarray(values)[:,None,...])
        write('non_tip_geom_name',np.asarray(calibration['non_tip_geom_names'],dtype=h5py.string_dtype()))
        # Dense fingertip poses in old bundles describe the old hand q.
        # Remove them rather than publishing stale FK for the new reference.
        for key in ('fingertip_pose_object','fingertip_twist_object'):
            if key in h:del h[key]
        h.attrs.update(healthy_reference_v1=True,healthy_reference_admissible=admissible,
            planner_bundle=True,planner_batch_size=1,finger_contact_plan=True,
            healthy_reference_source=str(Path(source).resolve()),
            healthy_reference_calibration_json=json.dumps(calibration),
            finger_contact_preload_m=0.,planned_surface_gap_m=calibration['config']['target_gap_m'],
            grasp_keyframe_valid_ratio=float(np.mean([s['valid'] for s in states])),
            finger_contact_feasible_ratio=float(np.mean([s['valid'] for s in states])),
            finger_contact_all_feasible=admissible,
            non_tip_clearance_m=calibration['config']['non_tip_clearance_m'],
            max_cross_finger_penetration_m=calibration['config']['max_cross_finger_penetration_m'],
            clearance_solver_sha256=hashlib.sha256(Path(__file__).with_name('healthy_grasp_planner.py').read_bytes()).hexdigest(),
            pad_normal_error_semantics='Opposing actual split-pad hull facet and source-mesh normals; diagnostic only',
            planning_semantics='Nonpenetrating kinematic reference; physical force/compliance are established by MCC',
            healthy_reference_palm_and_timing_unchanged=True,
            collection_admissible=admissible)
        np.testing.assert_array_equal(h['palm_pose_object'][:],original['palm_pose_object'][:])
        if 'palm_twist_object' in original:
            np.testing.assert_array_equal(h['palm_twist_object'][:],original['palm_twist_object'][:])


def plan_continuous(a,reference,c,started):
    values=[];states=[];previous=None;fallback_failures=[]
    for frame,pose in enumerate(reference.poses):
        if previous is None:
            q,state=c.solve(pose,reference.sample(0)[1],multistart=3)
        else:
            q,state=c.advance(pose,previous,a.max_reference_velocity_rad_s*reference.dt)
        prior=previous
        values.append(q);states.append(state);previous=q
        if not state['valid']:fallback_failures.append(frame)
        if not state['valid'] and not a.continue_after_planning_failure:
            # Reject the candidate rather than exporting a shortened path or
            # spending the remaining budget following an invalid state.
            failure=dict(frame=frame,q=q.tolist(),palm_pose_object=pose.tolist(),
                previous_q=prior.tolist() if prior is not None else None,**state_report(state))
            save(a.output/'failed_state.json',failure)
            summary=dict(input=str(a.input.resolve()),output=str(a.input.resolve()),completed=True,
                mode='bounded_continuation',collection_admissible=False,full_reference_completed=False,
                failed_frame=frame,checked_frames=frame+1,total_reference_frames=len(reference.poses),
                no_shortened_reference_exported=True,rounds=[],
                reason='No healthy continuation found within current search budget; not a proof of infeasibility',
                elapsed_s=time.monotonic()-started)
            save(a.output/'summary.json',summary)
            save(a.output/'status.json',dict(phase='rejected',**summary))
            print(f'[HEALTH-CONTINUOUS] rejected at frame={frame}; no shortened reference exported',flush=True)
            return
        if frame%100==0 or frame==len(reference.poses)-1:
            save(a.output/'status.json',dict(phase='continuous_contact_planning',
                completed_frames=frame+1,total_frames=len(reference.poses),
                accepted_frames=sum(s['valid'] for s in states),
                first_invalid_frame=fallback_failures[0] if fallback_failures else None,
                elapsed_s=time.monotonic()-started))
            print(f'[HEALTH-CONTINUOUS] {frame+1}/{len(reference.poses)} valid={sum(s["valid"] for s in states)}',flush=True)
    c.temporal_limits=None
    q=np.asarray(values);frames=np.arange(len(q))
    # Knots at every control frame make the collector's sampled smoothstep
    # exactly reproduce this continuation; no sparse interpolation is assumed.
    dense,valid,_,report=certify_dense(c,reference,frames,q,
        a.max_reference_velocity_rad_s,a.output/'continuous_round0.npz')
    admissible=bool(valid.all() and all(s['valid'] for s in states))
    export_path=a.output/('reference.h5' if admissible else 'reference_rejected.h5')
    export_reference(a.input,export_path,reference,frames,q,states,dense,admissible,c.calibration)
    save(a.output/'summary.json',dict(input=str(a.input.resolve()),output=str(export_path.resolve()),
        completed=True,mode='bounded_continuation',collection_admissible=admissible,
        rounds=[dict(round=0,knots=len(q),valid_knots=sum(s['valid'] for s in states),**report)],
        elapsed_s=time.monotonic()-started,initial=state_report(states[0]),
        palm_and_timing_unchanged=True,
        hand_planner_sha256=hashlib.sha256(Path(__file__).with_name('healthy_grasp_planner.py').read_bytes()).hexdigest(),
        qualification='Every serialized control frame; kinematic only, no physical force claim'))
    save(a.output/'status.json',dict(phase='completed',collection_admissible=admissible,
        completed_frames=len(q),elapsed_s=time.monotonic()-started))
    print('[HEALTH-CONTINUOUS] completed; admissible=',admissible,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bank',type=Path,default=DEFAULT_BANK)
    p.add_argument('--mode',choices=('continuous','sparse'),default='continuous',
        help='Continuous follows the previous healthy state at every control frame; sparse is diagnostic')
    p.add_argument('--continue-after-planning-failure',action='store_true',
        help='Diagnostic only: continue a rejected geometric search; never turns it into a valid reference')
    p.add_argument('--knot-stride',type=int,default=25)
    p.add_argument('--adaptive-rounds',type=int,default=2)
    p.add_argument('--insert-per-round',type=int,default=20)
    p.add_argument('--max-reference-velocity-rad-s',type=float,default=2.)
    a=p.parse_args()
    if min(a.knot_stride,a.insert_per_round,a.max_reference_velocity_rad_s)<=0 or a.adaptive_rounds<0:
        p.error('Invalid temporal bounds or refinement budget')
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'status.json').exists() or (a.output/'summary.json').exists():
        p.error('Output contains a previous run; choose a fresh directory')
    reference=PlanReference.load(a.input)
    c=TemporalHealthyPlanner(reference.object_id,reference.scale,a.bank)
    c.calibration['non_tip_geom_names']=list(c.geometry.non_tip_names)
    c.calibration['bank_path']=str(a.bank.resolve())
    c.calibration['bank_sha256']=hashlib.sha256(a.bank.read_bytes()).hexdigest()
    save(a.output/'calibration.json',c.calibration)
    started=time.monotonic()
    with h5py.File(a.input) as source:
        # A raw palm plan contains a dense, constant initial-q placeholder.
        # These placeholders are seeds, not thousands of required IK knots.
        source_knots=reference.frames if 'grasp_keyframe_q' in source else np.array([0])
    frames=np.unique(np.r_[source_knots,np.arange(0,len(reference.poses),a.knot_stride),len(reference.poses)-1])
    save(a.output/'protocol.json',dict(input=str(a.input.resolve()),
        input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
        planner_sha256=hashlib.sha256(Path(__file__).with_name('healthy_grasp_planner.py').read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bank_sha256=hashlib.sha256(a.bank.read_bytes()).hexdigest(),
        initial_knots=frames.tolist(),config=asdict(c.config),
        mode=a.mode,native_mujoco_version=mujoco.__version__,
        temporal_config=dict(knot_stride=a.knot_stride,adaptive_rounds=a.adaptive_rounds,
            insert_per_round=a.insert_per_round,max_reference_velocity_rad_s=a.max_reference_velocity_rad_s,
            continue_after_planning_failure=a.continue_after_planning_failure)))
    snapshot=a.output/'source';snapshot.mkdir()
    for name in ('replan_healthy_reference.py','healthy_grasp_planner.py','true_tip_waypoint_optimizer.py'):
        path=Path(__file__).with_name(name)
        (snapshot/name).write_bytes(path.read_bytes())
    if a.mode=='continuous':
        plan_continuous(a,reference,c,started)
        return
    saved={};rounds=[]
    for round_id in range(a.adaptive_rounds+1):
        q=[];states=[];previous=None;previous_frame=None
        for frame in frames:
            source=saved.get(int(frame),reference.sample(frame)[1])
            c.temporal_limits=None
            if previous is not None:
                # Smoothstep peak velocity is 1.5 times the endpoint secant.
                delta=a.max_reference_velocity_rad_s*(frame-previous_frame)*reference.dt/1.5
                c.temporal_limits=(previous-delta,previous+delta)
            value,state=c.solve(reference.poses[frame],source,previous,multistart=3)
            q.append(value);states.append(state);saved[int(frame)]=value
            previous=value;previous_frame=frame
            if len(q)%10==0 or len(q)==len(frames):
                save(a.output/'status.json',dict(phase='solving_knots',round=round_id,
                    completed_knots=len(q),total_knots=len(frames),
                    accepted_knots=sum(s['valid'] for s in states),elapsed_s=time.monotonic()-started))
                print(f'[HEALTH-PLAN] round={round_id} {len(q)}/{len(frames)} valid={sum(s["valid"] for s in states)}',flush=True)
        c.temporal_limits=None
        dense,valid,severity,report=certify_dense(c,reference,frames,q,a.max_reference_velocity_rad_s,
            a.output/f'continuous_round{round_id}.npz')
        all_knots=all(s['valid'] for s in states)
        rounds.append(dict(round=round_id,knots=len(frames),valid_knots=sum(s['valid'] for s in states),**report))
        admissible=bool(all_knots and valid.all())
        export_path=a.output/('reference.h5' if admissible else 'reference_rejected.h5')
        export_reference(a.input,export_path,reference,frames,q,states,dense,admissible,c.calibration)
        save(a.output/'summary.json',dict(input=str(a.input.resolve()),output=str(export_path.resolve()),
            completed=True,collection_admissible=admissible,rounds=rounds,elapsed_s=time.monotonic()-started,
            hand_planner_sha256=hashlib.sha256(Path(__file__).with_name('healthy_grasp_planner.py').read_bytes()).hexdigest(),
            initial=state_report(states[0]),palm_and_timing_unchanged=True,
            qualification='Every raw knot and every interpolated frame; no skipping invalid nodes; kinematic only'))
        if admissible or round_id==a.adaptive_rounds or not all_knots:break
        # One worst interior violation per existing interval, with finite
        # refinement budget. No trajectory truncation or endpoint replacement.
        candidates=[]
        for left,right in zip(frames[:-1],frames[1:]):
            interior=np.arange(left+1,right)
            if len(interior) and (~valid[interior]).any():
                k=interior[np.argmax(severity[interior])]
                candidates.append(int(k))
        candidates=sorted(candidates,key=lambda k:severity[k],reverse=True)[:a.insert_per_round]
        if not candidates:break
        frames=np.unique(np.r_[frames,candidates])
    save(a.output/'status.json',dict(phase='completed',collection_admissible=admissible,
        elapsed_s=time.monotonic()-started,knots=len(frames)))
    print('[HEALTH-PLAN] completed; admissible=',admissible,flush=True)


if __name__=='__main__':main()
