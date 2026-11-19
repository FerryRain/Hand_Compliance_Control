"""Plan spatial exploration references, time them by speed, refine teachers.

Four path families; paired speeds use identical geometry/object/clearance.
Output H5s are planning candidates. Only subsequent measured collection may
provide DP supervision. No short-interval extraction or duration padding.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from generate_manifold_palm_plan import (palm_outline_clearance_stats,
    resample_pose_uniform_arc_length, pose_to_rt, palm_plan_motion_metrics)
from object_catalog import load_object_config, MeshNormalOracle
from replan_grasp_221 import continuous_geometry, sha256, write_json

SCRIPTS = Path(__file__).resolve().parent
DATA = SCRIPTS.parent / 'data'
FAMILIES = ('meridian_arc', 'oblique_orbit', 'spatial_spiral', 'direction_sweep')


def candidate(index, args):
    group, speed_id = divmod(index, len(args.speeds_m_s))
    family = FAMILIES[group % len(FAMILIES)]
    rng = np.random.default_rng(args.seed + group)
    row = dict(candidate_id=index, geometry_group=group, family=family,
        object_seed=800012+group, clearance_seed=args.seed+group,
        target_speed_m_s=float(args.speeds_m_s[speed_id]), direction=int(rng.choice([-1,1])),
        angle_deg=float(rng.uniform(160,220)), meridian_deg=float(rng.uniform(0,360)),
        section_fraction=-1., tilt_deg=0., azimuth_deg=float(rng.uniform(0,360)),
        meridian_sweep_deg=0., meridian_weave_deg=0., palm_yaw_deg=0.)
    if family == 'oblique_orbit':
        row.update(angle_deg=float(rng.uniform(330,390)),section_fraction=float(rng.uniform(.3,.75)),
                   tilt_deg=float(rng.uniform(20,50)))
    elif family == 'spatial_spiral':
        row.update(meridian_sweep_deg=float(rng.uniform(300,420))*row['direction'])
    elif family == 'direction_sweep':
        row.update(angle_deg=float(rng.uniform(300,360)),section_fraction=float(rng.uniform(.3,.7)),
                   tilt_deg=float(rng.uniform(30,60)),meridian_weave_deg=35.)
    elif group == 0:
        # First canary reuses the already visualized long39 path parameters.
        row.update(angle_deg=179.21977637074366,meridian_deg=217.3421463731595,
                   direction=1,palm_yaw_deg=15.545562022485164)
    return row


def time_by_speed(source, output, speed_m_s, max_wrist_deg_s=60.):
    if speed_m_s <= 0 or not np.isfinite(speed_m_s):
        raise ValueError('target mean speed must be finite and positive')
    with h5py.File(source) as original:
        pose=np.asarray(original['palm_pose_object'])[:,0]
        dt=float(original.attrs.get('control_dt',.01))
        arc=float(np.linalg.norm(np.diff(pose[:,:3],axis=0),axis=-1).sum())
        frames=max(160,int(np.ceil(arc/(speed_m_s*dt)))+1)
        initial_frames=frames
        for _ in range(5):
            new_pose=resample_pose_uniform_arc_length(pose,frames,ramp_fraction=.10)
            rotations=Rotation.from_quat(new_pose[:,[4,5,6,3]])
            angular=(rotations[1:]*rotations[:-1].inv()).as_rotvec()/dt
            peak=float(np.degrees(np.linalg.norm(angular,axis=-1).max()))
            if max_wrist_deg_s <= 0 or peak <= max_wrist_deg_s*1.001: break
            frames=int(np.ceil(frames*peak/max_wrist_deg_s*1.01))
        else:
            raise ValueError('wrist speed cap did not converge')
        config=load_object_config(str(original.attrs['object_id']))
        oracle=MeshNormalOracle.from_config(config,scale=float(original.attrs.get('object_scale',1.)))
        pos,rot=pose_to_rt(new_pose)
        stats=np.asarray([palm_outline_clearance_stats(p,r,oracle)[:2] for p,r in zip(pos,rot)])
        target=float(original.attrs['planner_palm_outline_clearance_m'])
        if not np.isfinite(stats).all() or stats[:,1].min()<target-1e-5:
            raise ValueError('speed resampling violates palm outline margin')
        twist=np.zeros((frames,6),np.float32)
        twist[1:,:3]=np.diff(new_pose[:,:3],axis=0)/dt;twist[1:,3:]=angular
        translation=np.linalg.norm(twist[1:,:3],axis=-1)
        realized_arc=float(np.linalg.norm(np.diff(new_pose[:,:3],axis=0),axis=-1).sum())
        output.parent.mkdir(parents=True,exist_ok=True)
        with h5py.File(output,'w') as target_file:
            replaced={'palm_pose_object','palm_twist_object','record_step','qvel',
                'planner_palm_outline_min_clearance_object','planner_palm_outline_mean_clearance_object'}
            for name,value in original.items():
                if name in replaced: continue
                data=value[:]
                if data.shape[0]==len(pose):
                    # Raw generator fields are static initial-state placeholders.
                    # Do not retime an optimized/measured trajectory this way.
                    if name.startswith('grasp_') or (name=='q_hand' and np.ptp(data,axis=0).max()>1e-7):
                        raise ValueError('time_by_speed expects a raw palm-only plan')
                    data=np.repeat(data[:1],frames,axis=0)
                target_file[name]=data
            target_file['palm_pose_object']=new_pose[:,None]
            target_file['palm_twist_object']=twist[:,None]
            target_file['record_step']=np.arange(frames)[:,None]
            target_file['qvel']=np.zeros((frames,1,16),np.float32)
            target_file['planner_palm_outline_min_clearance_object']=stats[:,1,None]
            target_file['planner_palm_outline_mean_clearance_object']=stats[:,0,None]
            for key,value in original.attrs.items():target_file.attrs[key]=value
            for key,value in palm_plan_motion_metrics(new_pose).items():
                target_file.attrs['planner_realized_'+key]=float(value)
            target_file.attrs.update(exploration_target_mean_speed_m_s=speed_m_s,
                exploration_realized_mean_speed_m_s=realized_arc/((frames-1)*dt),
                exploration_peak_speed_m_s=float(translation.max()),
                exploration_wrist_speed_cap_deg_s=max_wrist_deg_s,
                exploration_peak_wrist_speed_deg_s=peak,exploration_frames=frames,
                exploration_time_semantics='path length / target mean speed, 10% end ramps; wrist cap may extend duration',
                exploration_supervision='initial-state placeholders; collect actual teacher before DP training')
        return dict(frames=frames,duration_s=(frames-1)*dt,path_length_m=realized_arc,
            target_mean_speed_m_s=speed_m_s,realized_mean_speed_m_s=realized_arc/((frames-1)*dt),
            peak_speed_m_s=float(translation.max()),peak_wrist_speed_deg_s=peak,
            duration_extended_for_wrist_cap=frames>initial_frames,
            min_palm_outline_clearance_m=float(stats[:,1].min()))


def repair_frames(path, existing, count=24):
    with np.load(path) as z:
        valid=z['valid'];non=z['non_tip_distance'];pad=z['pad_distance']
        margin=float(z['non_tip_clearance_m']) if 'non_tip_clearance_m' in z else .002
        severity=np.maximum(margin-non.min(-1),0)/margin
        severity+=np.maximum(pad.max(-1)-.004,0)/.004 + np.maximum(-.012-pad.min(-1),0)/.012
    candidates=np.flatnonzero(~valid)
    available=[int(i) for i in candidates[np.argsort(-severity[candidates])] if int(i) not in existing]
    chosen=[]
    for frame in available:
        if all(abs(frame-x)>=3 for x in chosen):chosen.append(frame)
        if len(chosen)>=count:break
    return chosen


def collection_admission(path):
    """Allow physical validation of reserve dips, never geometric collisions.

    Passing this check is not acceptance as training data. Real force/contact,
    complete execution and the independent stop are checked by collection.
    """
    with np.load(path) as z:
        if 'cut_cap' not in z:
            return False  # Older diagnostics need a full recheck.
        non=z['non_tip_distance'];pad=z['pad_distance'];depth=z['cross_finger_penetration']
        valid=(np.isfinite(non).all(-1)&np.isfinite(pad).all(-1)&np.isfinite(depth)
            &(non.min(-1)>=0)&(pad.min(-1)>=-.012)&(pad.max(-1)<=.004)
            &(depth<=.0001+1e-6)&~z['cut_cap'].any(-1))
        return bool(valid.all())


def outward_detour(source, output, centers, amplitude_m=.004, sigma_frames=100):
    """Small smooth palm-reference repair; preserves the full route and timing."""
    if amplitude_m <= 0 or amplitude_m > .01 or sigma_frames < 10:
        raise ValueError('detour requires 0..10mm and >=10 frame smoothing')
    with h5py.File(source) as handle:
        if 'grasp_keyframe_q' in handle:
            raise ValueError('repair the palm-only input, then solve a new teacher')
        pose=handle['palm_pose_object'][:,0];dt=float(handle.attrs['control_dt'])
        positions,rotations=pose_to_rt(pose)
        t=np.arange(len(pose));centers=sorted(set(map(int,centers)))
        if not centers or min(centers)<0 or max(centers)>=len(pose):
            raise ValueError('detour centers must be inside the path')
        offset=sum(np.exp(-.5*((t-c)/sigma_frames)**2)for c in centers)
        offset*=amplitude_m/offset.max()
        pose[:,:3]+=offset[:,None]*rotations[:,:,2]
        oracle=MeshNormalOracle.from_config(load_object_config(str(handle.attrs['object_id'])),
            scale=float(handle.attrs.get('object_scale',1.)))
        p,r=pose_to_rt(pose)
        stats=np.array([palm_outline_clearance_stats(x,y,oracle)[:2]for x,y in zip(p,r)])
        if stats[:,1].min()<float(handle.attrs['planner_palm_outline_clearance_m'])-1e-5:
            raise ValueError('outward repair failed the complete-palm outline check')
    rotation=Rotation.from_quat(pose[:,[4,5,6,3]])
    twist=np.zeros((len(pose),6),np.float32)
    twist[1:,:3]=np.diff(pose[:,:3],axis=0)/dt
    twist[1:,3:]=(rotation[1:]*rotation[:-1].inv()).as_rotvec()/dt
    shutil.copyfile(source,output)
    with h5py.File(output,'r+') as handle:
        handle['palm_pose_object'][:,0]=pose;handle['palm_twist_object'][:,0]=twist
        handle['planner_palm_outline_min_clearance_object'][:,0]=stats[:,1]
        handle['planner_palm_outline_mean_clearance_object'][:,0]=stats[:,0]
        for key,value in palm_plan_motion_metrics(pose).items():handle.attrs['planner_realized_'+key]=float(value)
        handle.attrs.update(local_detour_centers_frames=centers,local_detour_max_m=amplitude_m,
            local_detour_sigma_frames=sigma_frames,local_detour_source=str(Path(source).resolve()),
            exploration_realized_mean_speed_m_s=float(np.linalg.norm(np.diff(pose[:,:3],axis=0),axis=-1).sum()/((len(pose)-1)*dt)))
    return dict(centers_frames=centers,max_outward_m=amplitude_m,sigma_frames=sigma_frames,
                preserves_entire_route=True,actual_plan=str(Path(output).resolve()))


def attempt(row,args):
    folder=args.output/f"candidate_{row['candidate_id']:04d}";folder.mkdir(exist_ok=True)
    result_path=folder/'result.json'
    if result_path.exists():return json.loads(result_path.read_text())
    raw=folder/'geometry.h5';timed=folder/'palm.h5'
    result=dict(row,passed=False,collection_candidate=False,folder=str(folder.resolve()))
    started=time.monotonic()
    command=[sys.executable,str(SCRIPTS/'generate_manifold_palm_plan.py'),
        '--output',str(raw),'--object-id','ycb_mustard','--frames','2400',
        '--path-mode','minimum_enclosing_ellipse','--angle-deg',str(row['angle_deg']),
        '--direction',str(row['direction']),'--ellipse-section-fraction',str(row['section_fraction']),
        '--ellipse-plane-tilt-deg',str(row['tilt_deg']),'--ellipse-azimuth-deg',str(row['azimuth_deg']),
        '--ellipse-meridian-deg',str(row['meridian_deg']),'--global-ellipse-plane-mode','long_axis',
        '--palm-tangent-sign','-1','--object-rotation-mode','uniform_so3','--seed',str(row['object_seed']),
        '--time-parameterization','arc_length','--palm-yaw-offset-deg',str(row['palm_yaw_deg']),
        '--palm-clearance-range-m',*map(str,args.palm_clearance_range_m),
        '--palm-clearance-seed',str(row['clearance_seed']),
        '--meridian-sweep-deg',str(row['meridian_sweep_deg']),
        '--meridian-weave-deg',str(row['meridian_weave_deg'])]
    result['generation_command']=command
    try:
        print(f'[EXPLORE] {row["candidate_id"]}: {row["family"]} v={row["target_speed_m_s"]}',flush=True)
        with (folder/'planning.log').open('w') as log:
            if args.palm_source is not None:
                source_folder=args.palm_source/f"candidate_{row['candidate_id']:04d}"
                saved=json.loads((source_folder/'result.json').read_text())
                if saved['status']!='palm_candidate' or any(saved[k]!=v for k,v in row.items()):
                    raise ValueError('source palm identity/config differs from requested candidate')
                shutil.copyfile(source_folder/'palm.h5',timed)
                result['timing']=saved['timing']
                result.pop('generation_command')
                result.update(source_palm=str((source_folder/'palm.h5').resolve()),
                    source_palm_sha256=sha256(source_folder/'palm.h5'))
            else:
                process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if process.returncode:
                    result.update(status='palm_generation_error',return_code=process.returncode)
                    return result
                result['timing']=time_by_speed(raw,timed,row['target_speed_m_s'],args.max_wrist_speed_deg_s)
            with h5py.File(timed,'r+') as handle:
                handle.attrs['exploration_family']=row['family'];handle.attrs['exploration_geometry_group']=row['geometry_group']
            if args.stage=='palm':
                result.update(status='palm_candidate',palm_plan=str(timed.resolve()))
                return result
            if args.hand_planner=='healthy':
                # Nonpenetrating references and explicit joint/motion reserve
                # are certified on every frame. Legacy preload tolerance may
                # not silently admit a rejected healthy reference.
                healthy_output=folder/'healthy_reference'
                optimize=[sys.executable,str(SCRIPTS/'replan_healthy_reference.py'),
                    '--input',str(timed),'--output',str(healthy_output),
                    '--bank',str(args.reference_bank.resolve()),'--knot-stride',str(args.knot_stride),
                    '--adaptive-rounds',str(args.repair_rounds),
                    '--insert-per-round',str(args.max_repair_knots)]
                result['hand_planning_command']=optimize
                process=subprocess.run(optimize,stdout=log,stderr=subprocess.STDOUT)
                if process.returncode:
                    result.update(status='healthy_hand_optimization_error',return_code=process.returncode)
                    return result
                certification=json.loads((healthy_output/'summary.json').read_text())
                accepted=bool(certification['collection_admissible'])
                result.update(plan=certification['output'],rounds=certification['rounds'],
                    passed=accepted,collection_candidate=accepted,
                    hand_planner='healthy',healthy_reference_summary=str((healthy_output/'summary.json').resolve()),
                    status='healthy_reference_candidate' if accepted else 'healthy_reference_rejected')
                if accepted:result['collection_plan']=certification['output']
                return result
            extra=[];rounds=[];detours=[]
            for round_id in range(args.repair_rounds+1):
                optimized=folder/f'teacher_round{round_id}.h5'
                optimize=[sys.executable,str(SCRIPTS/'optimize_contact_plan.py'),
                    '--input',str(timed),'--output',str(optimized),'--true-tip-waypoints','--keyframes-only',
                    '--reference-bank',str(args.reference_bank.resolve()),'--knot-stride',str(args.knot_stride),
                    '--temporal-seed-first','--ik-posture-gain','.18','--non-tip-clearance-m','.002',
                    '--max-cross-finger-penetration-m','.0001','--max-posture-deviation-rad','.90',
                    '--no-extract-feasible-segment']
                optimize+=['--non-tip-preferred-clearance-m',str(args.non_tip_preferred_clearance_m)]
                if extra:optimize+=['--extra-keyframe-frames',*map(str,sorted(extra))]
                process=subprocess.run(optimize,stdout=log,stderr=subprocess.STDOUT)
                if process.returncode:
                    result.update(status='teacher_optimization_error',return_code=process.returncode)
                    return result
                with h5py.File(optimized) as handle:
                    valid=np.asarray(handle['grasp_keyframe_valid']).reshape(-1).astype(bool)
                    indices=np.asarray(handle['grasp_keyframe_frame_index']).reshape(-1)
                diag=continuous_geometry(optimized,folder/f'geometry_round{round_id}.npz')
                rounds.append(dict(round=round_id,keyframes=len(valid),valid_keyframes=int(valid.sum()),continuous_geometry=diag))
                result.update(plan=str(optimized.resolve()),rounds=rounds)
                if valid.all() and collection_admission(folder/f'geometry_round{round_id}.npz'):
                    result.update(collection_candidate=True,collection_plan=str(optimized.resolve()))
                if valid.all() and diag['passed']:
                    result.update(status='teacher_geometry_candidate',passed=True);break
                if round_id==args.repair_rounds:
                    result['status']='teacher_geometry_rejected';break
                if not valid.all():
                    with h5py.File(optimized) as handle:
                        non_tip=np.asarray(handle['grasp_keyframe_non_tip_distance'])
                    centers=indices[(~valid)&(non_tip.min(-1)<.002-1e-6)]
                    if len(centers) and len(detours)<args.palm_detour_rounds:
                        repaired=folder/f'palm_detour{len(detours)}.h5'
                        detours.append(outward_detour(timed,repaired,centers,
                            amplitude_m=args.palm_detour_max_m,
                            sigma_frames=max(10,int(.03/(row['target_speed_m_s']*.01)))))
                        timed=repaired;result['palm_detours']=detours
                        continue
                    result['status']='teacher_geometry_rejected';break
                added=repair_frames(folder/f'geometry_round{round_id}.npz',set(indices.tolist()),args.max_repair_knots)
                if not added:
                    result['status']='teacher_geometry_rejected';break
                extra=sorted(set(extra+added))
            if result.get('collection_candidate') and not result['passed']:
                result['status']='physical_validation_candidate_with_reserve_dips'
    except Exception as error:
        result.update(status='error',error=f'{type(error).__name__}: {error}')
    finally:
        result['elapsed_s']=time.monotonic()-started;write_json(result_path,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidates',type=int,default=221)
    parser.add_argument('--candidate-ids',type=int,nargs='+',help='Explicit IDs for representative pilot checks')
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--stage',choices=['palm','teacher'],default='teacher')
    parser.add_argument('--palm-source',type=Path,help='Reuse previously generated candidates from a palm stage')
    parser.add_argument('--speeds-m-s',type=float,nargs='+',default=[.03,.05])
    parser.add_argument('--max-wrist-speed-deg-s',type=float,default=60.)
    parser.add_argument('--palm-clearance-range-m',type=float,nargs=2,default=[.03,.04])
    parser.add_argument('--seed',type=int,default=20261015)
    parser.add_argument('--reference-bank',type=Path,default=DATA/'dp0_true_tip_1500/long_chains/training_only_healthy_seed_bank.npz')
    parser.add_argument('--hand-planner',choices=['healthy','legacy'],default='healthy',
        help='Healthy uses no planned preload, explicit joint reserve and every-frame certification')
    parser.add_argument('--knot-stride',type=int,default=25)
    parser.add_argument('--non-tip-preferred-clearance-m',type=float,default=.005,
        help='Soft reserve; hard acceptance remains 2mm and actual 100N safety is independent')
    parser.add_argument('--repair-rounds',type=int,default=2)
    parser.add_argument('--max-repair-knots',type=int,default=24)
    parser.add_argument('--palm-detour-rounds',type=int,default=1)
    parser.add_argument('--palm-detour-max-m',type=float,default=.004)
    args=parser.parse_args()
    if args.candidates<1 or args.workers<1 or args.start<0 or args.knot_stride<1 or args.repair_rounds<0 or args.max_repair_knots<1:
        parser.error('invalid count, workers, stride or repair budget')
    if not all(np.isfinite(v) and v>0 for v in args.speeds_m_s):parser.error('positive speeds required')
    if args.candidate_ids is not None and (min(args.candidate_ids)<0 or len(set(args.candidate_ids))!=len(args.candidate_ids)):
        parser.error('candidate IDs must be unique and nonnegative')
    if args.palm_source is not None and args.stage!='teacher':parser.error('palm-source requires teacher stage')
    if args.palm_detour_rounds<0 or not 0<args.palm_detour_max_m<=.01:
        parser.error('invalid detour count or amplitude')
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()},
        scripts={name:sha256(SCRIPTS/name)for name in ['plan_exploration_dataset.py','generate_manifold_palm_plan.py',
            'optimize_contact_plan.py','true_tip_waypoint_optimizer.py','replan_grasp_221.py',
            'healthy_grasp_planner.py','replan_healthy_reference.py']},
        scope='offline teacher candidates; no physical collection or DP training')
    path=args.output/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('resume protocol changed; use fresh output')
    write_json(path,protocol)
    rows=[];started=time.monotonic()
    ids=args.candidate_ids if args.candidate_ids is not None else list(range(args.start,args.start+args.candidates))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(attempt,candidate(i,args),args)for i in ids]
        for future in as_completed(futures):
            result=future.result();rows.append(result);rows.sort(key=lambda r:r['candidate_id'])
            summary=dict(requested=len(ids),completed=len(rows),qualified=sum(r['passed']for r in rows),
                collection_candidates=sum(r.get('collection_candidate',False)for r in rows),
                stage=args.stage,elapsed_s=time.monotonic()-started,rows=rows)
            write_json(args.output/'progress.json',summary)
            (args.output/'passing_plans.txt').write_text(''.join(r['plan']+'\n'for r in rows if r['passed']))
            (args.output/'collection_candidates.txt').write_text(''.join(r['collection_plan']+'\n'
                for r in rows if r.get('collection_candidate')))
            print(f'[EXPLORE] {len(rows)}/{len(ids)}; candidate {result["candidate_id"]}: '+result['status'],flush=True)
    write_json(args.output/'summary.json',summary)


if __name__=='__main__':main()
