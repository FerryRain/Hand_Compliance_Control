"""Long palm-reference stress tests without future teacher hand trajectories.

Retain failed hand-screening plans. Only palm-input validity and a physically
loaded initial grasp gate task interpretation. No resets during exploration.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
from dataclasses import replace
import h5py,numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'mcc_finger_compliance_control/data'
PLANS=DATA/'dp0_true_tip_1500/long_chains/plans'
DP=DATA/'DPPO/DP0_zero_residual_canary_20261012/candidate.pt'
RESIDUAL=DATA/'RL0/ppo_DP0_mcc_retrained_20261011/best.pt'
PROFILE=DATA/'contact_projection_full_development_20261011/selected_mcc_profile.json'
REGIONS=('bottom','lower_body','upper_body','shoulder_neck','cap')
EDGES=np.array([0,.10,.60,.85,.93,1.000001])


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def prepare(output):
    old=json.loads((PLANS/'summary.json').read_text());rows=[]
    for index,item in enumerate(sorted(old['rows'],key=lambda r:r['name'])):
        raw=PLANS/(item['name']+'.h5');opt=PLANS/(item['name']+'_opt.h5')
        row=dict(trajectory_id=index,name=item['name'],family=item['family'],palm_path=str(raw),
            old_teacher_passed=item['passed'],old_return_code=item['return_code'])
        if not raw.exists():
            row['status']='no_palm_reference';row['reason']='Palm generation failed or source file is absent'
            log=Path(item['log'])
            if log.exists():row['planner_error_tail']=log.read_text(errors='replace').splitlines()[-4:]
            rows.append(row);continue
        with h5py.File(raw) as f:
            poses=f['palm_pose_object'][:,0].astype(np.float64);dt=float(f.attrs['control_dt'])
            linear=np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=-1)/dt
            rot=Rotation.from_quat(poses[:,[4,5,6,3]])
            angular=(rot[1:]*rot[:-1].inv()).magnitude()/dt
            clearance=np.asarray(f['planner_palm_outline_min_clearance_object'])
            q0=f['q_hand'][0,0].tolist()
        issues=[]
        if not np.isfinite(poses).all() or np.max(np.abs(np.linalg.norm(poses[:,3:],axis=-1)-1))>1e-3:issues.append('invalid_pose')
        if abs(dt-.01)>1e-9:issues.append('unexpected_control_dt')
        if linear.max()>.08 or angular.max()>np.pi/2:issues.append('exceeds_test_motion_limits')
        if np.isfinite(clearance).any() and np.nanmin(clearance)<-.001:issues.append('palm_outline_penetration')
        row.update(frames=len(poses),duration_s=len(poses)*dt,path_length_m=float(linear.sum()*dt),
            max_speed_mm_s=float(linear.max()*1000),max_angular_deg_s=float(angular.max()*180/np.pi),
            min_outline_clearance_m=float(np.nanmin(clearance)),initial_q=q0,initial_q_source='raw_default')
        if opt.exists():
            with h5py.File(opt) as f:
                if not np.array_equal(f['palm_pose_object'][:],poses.astype(np.float32)[:,None]):
                    raise ValueError(f'Hand optimization changed palm reference: {opt}')
                row['initial_q']=f['grasp_keyframe_q'][0].tolist()
                row['initial_q_source']='best_first_keyframe_only'
                row['initial_teacher_valid']=bool(f['grasp_keyframe_valid'][0])
                row['old_valid_keyframes']=int(np.asarray(f['grasp_keyframe_valid']).sum())
        row['status']='palm_input_invalid' if issues else 'ready'
        row['issues']=issues;rows.append(row)
    manifest=dict(rows=rows,recorded_candidates=len(rows),existing_palm_references=sum(Path(r['palm_path']).exists() for r in rows),
        deployable_palm_references=sum(r['status']=='ready' for r in rows),
        admission='Ignore all future hand/keyframe screening results; only input motion validity gates admission',
        initial_state='Use the best initial q only, regardless of old teacher screening label; physical initialization is checked separately',
        motion_limits='80mm/s and90deg/s are test protocol limits, not a robot-arm workspace/safety certification',
        initialization_hold_frames=100,settle_frames=100,
        future_hand='No teacher q after initialization; the q_hand adapter array is constant q0 for all frames',
        dp=str(DP),residual=str(RESIDUAL),residual_status='Historical unaccepted R0 candidate, used only as diagnostic comparison',
        physics='Frozen true-tip geometry, accepted MCC profile and100N net-force stop')
    write(output/'manifest.json',manifest);return manifest


def pack(rows,initialization_frames):
    from palm_planner_features import future_palm_delta_pose_palm
    pose=[];twist=[];q=[];commands=[]
    durations={r['frames'] for r in rows}
    if len(durations)!=1:raise ValueError('Batch only equal-duration references; never pad short trajectories')
    for row in rows:
        with h5py.File(row['palm_path']) as f:
            p=f['palm_pose_object'][:,0];v=f['palm_twist_object'][:,0]
        p=np.concatenate((np.repeat(p[:1],initialization_frames,axis=0),p))
        v=np.concatenate((np.zeros((initialization_frames,6),np.float32),v))
        q.append(np.repeat(np.asarray(row['initial_q'],np.float32)[None],len(p),axis=0))
        pose.append(p);twist.append(v)
        # Encode separately per lane: flattened interleaved environments must
        # never share an episode ID and accidentally cross lane boundaries.
        commands.append(future_palm_delta_pose_palm(p,np.zeros(len(p),np.int64),step_frames=20))
    return dict(q_hand=np.stack(q,axis=1),palm_pose_object=np.stack(pose,axis=1),
        palm_twist_object=np.stack(twist,axis=1),planner_palm_delta_pose_palm_commanded=np.stack(commands,axis=2))


def longest_loss(valid):
    run=longest=0;first=None
    for t,ok in enumerate(valid):
        run=0 if ok else run+1;longest=max(longest,run)
        if run==6 and first is None:first=t-5
    return longest,first


def summarize_lane(row,values,meta,lane,hold,oracle):
    strict=values['strict'][hold:,lane];found=values['found'][hold:,lane]
    seen=values['seen'][hold:,lane];all4=strict.all(-1)
    stop=meta['per_env_stop_frame'][lane]
    initial_bad=meta['per_env_initial_rejected'][lane] or (0<=stop<hold)
    completed=int(seen.sum());last=max(0,min(completed-1,row['frames']-1))
    with h5py.File(row['palm_path']) as f:reference=f['palm_pose_object'][:,0]
    arc=np.concatenate(([0.],np.cumsum(np.linalg.norm(np.diff(reference[:,:3],axis=0),axis=-1))))
    longest,first=longest_loss(all4)
    point=values['points'][hold:,lane];contacts=point[strict]
    center=oracle.vertices.mean(0);_,_,axes=np.linalg.svd(oracle.vertices-center,full_matrices=False)
    axis=axes[0];axis*=1 if axis[np.abs(axis).argmax()]>0 else -1
    height=(oracle.vertices-center)@axis;low,high=height.min(),height.max()
    def region(points):
        fraction=np.clip(((points-center)@axis-low)/(high-low),0,1)
        return np.clip(np.searchsorted(EDGES,fraction,side='right')-1,0,4)
    # Area of source triangles grouped into10mm cells. This estimates visited
    # patches, not exact contact-patch area or a geodesic traveled distance.
    origin=oracle.mesh.bounds[0];size=.01
    face_cells=np.floor((oracle.face_centroids-origin)/size).astype(np.int64)
    unique,inverse=np.unique(face_cells,axis=0,return_inverse=True)
    weights=np.bincount(inverse,weights=oracle.mesh.area_faces)
    lookup={tuple(cell):i for i,cell in enumerate(unique)}
    visited=set();visited4=set()
    for points,target in [(contacts,visited),(point[all4].reshape(-1,3),visited4)]:
        if len(points):
            _,indices=oracle.face_tree.query(points)
            target.update(int(inverse[i]) for i in np.asarray(indices).reshape(-1))
    area=float(weights.sum());rindex=region(oracle.face_centroids)
    area_by_region=[float(oracle.mesh.area_faces[rindex==i].sum()) for i in range(5)]
    counts=region(contacts) if len(contacts) else np.array([],int)
    initial_rates=values['strict'][hold-20:hold,lane].mean(0)
    result=dict(trajectory_id=row['trajectory_id'],name=row['name'],family=row['family'],initial_teacher_valid=row.get('initial_teacher_valid'),
        initial_physically_valid=not initial_bad,initial_strict_per_finger=initial_rates.tolist(),
        status='initialization_failed' if initial_bad else ('force_stop' if stop>=0 else 'reference_completed'),
        stop_frame=stop,stop_task_frame=max(0,stop-hold) if stop>=0 else None,
        completed_reference_frames=0 if initial_bad else completed,reference_frames=row['frames'],
        reference_arc_fraction=0. if initial_bad else float(arc[last]/max(arc[-1],1e-12)),
        healthy_arc_fraction=float((np.diff(arc,prepend=0)*all4).sum()/max(arc[-1],1e-12)),
        strict_healthy_four_rate=float(all4.mean()),strict_healthy_per_finger=strict.mean(0).tolist(),
        found_four_rate=float(found.all(-1).mean()),found_per_finger=found.mean(0).tolist(),
        longest_four_loss_frames=longest,first_loss_over5_frame=first,
        peak_force_n=float(values['force'][:,lane].max()),
        visited_surface_cell_area_fraction=float(weights[list(visited)].sum()/area) if visited else 0.,
        four_contact_surface_cell_area_fraction=float(weights[list(visited4)].sum()/area) if visited4 else 0.,
        visited_surface_cells=len(visited),surface_regions={REGIONS[i]:dict(valid_contact_samples=int((counts==i).sum()),
            mesh_region_area_m2=area_by_region[i]) for i in range(5)},
        reference_region_sequence=[],region_transition_windows=[])
    control=reference[:,:3]+Rotation.from_quat(reference[:,[4,5,6,3]]).apply([-.0559703,-.04142053,-.0340008])
    _,nearest=oracle.face_tree.query(control);reference_regions=region(oracle.face_centroids[nearest])
    transitions=np.flatnonzero(reference_regions[1:]!=reference_regions[:-1])+1
    # Suppress coarse triangle-boundary chatter in qualitative event windows.
    selected=[]
    for t in transitions:
        if not selected or t-selected[-1]>=100:selected.append(int(t))
    result['reference_region_sequence']=[REGIONS[int(reference_regions[0])]]+[REGIONS[int(reference_regions[t])] for t in selected]
    for t in selected:
        lo=max(0,t-100);hi=min(len(all4),t+100)
        result['region_transition_windows'].append(dict(frame=t,from_region=REGIONS[int(reference_regions[t-1])],to_region=REGIONS[int(reference_regions[t])],
            strict=float(all4[lo:hi].mean()),found=float(found[lo:hi].all(-1).mean()),executed=bool(seen[t])))
    if first is not None and seen[first]:
        f=first+hold
        result['first_loss_diagnostic']=dict(q_live=values['q'][f,lane].tolist(),dp_nominal=values['dp'][f,lane].tolist(),
            residual=values['offset'][f,lane].tolist(),pre_mcc=values['pre_mcc'][f,lane].tolist(),mcc_command=values['command'][f,lane].tolist(),
            pad_force=values['pad_force'][f,lane].tolist(),connector_force=values['connector_force'][f,lane].tolist())
    return result


def worker(args,manifest):
    import torch
    import deploy_b_gpu_serial_parity as deployment
    from batched_b_control_chain import BatchedBControlChain
    from true_fingertip_contact import ContactSlotObserver
    from mcc_force_profiles import configured_mcc,attach_force_profile
    from cuda_graph_hand_fk import CUDAGraphHandFK
    from cuda_graph_control_kernels import ControlGraphKernels
    from async_dp_planner import AsyncDPPlanner
    from train_rl0 import RL0Residual
    from residual_ppo import GaussianActor
    from object_catalog import MeshNormalOracle,load_object_config
    rows=[r for r in manifest['rows'] if r['trajectory_id'] in args.ids]
    hold=100;data=pack(rows,hold);T=len(data['q_hand']);B=len(rows);device='cuda:0'
    observer=ContactSlotObserver(controller_scope='pad',capture_collision=True)
    checkpoint=torch.load(args.residual,map_location='cpu',weights_only=False)
    cfg=dict(checkpoint['config']);cfg['frames']=T;bounds=checkpoint['bounds']
    class Agent:
        def __init__(self):
            self.actor=GaussianActor(528,cfg['hidden'],torch.full((16,),.1,device=device)).to(device).eval()
            self.actor.load_state_dict(checkpoint['actor']);self.actor.requires_grad_(False)
        def act(self,obs,zero=False,evaluate=True):
            self.last_z=self.actor.net(obs) if not zero else torch.zeros(len(obs),16,device=device)
            self.last_logp=torch.zeros(len(obs),device=device)
            return self.last_z.tanh()
    agent=Agent();profile=json.loads(PROFILE.read_text())
    arrays={name:torch.zeros((T,B,*shape),device=device,dtype=dtype) for name,shape,dtype in [
        ('strict',(4,),torch.bool),('found',(4,),torch.bool),('seen',(),torch.bool),
        ('force',(),torch.float32),('q',(16,),torch.float32),('dp',(16,),torch.float32),
        ('pre_mcc',(16,),torch.float32),('command',(16,),torch.float32),('offset',(16,),torch.float32),
        ('points',(4,3),torch.float32),('pad_force',(4,3),torch.float32),('connector_force',(4,3),torch.float32)]}
    class Residual(RL0Residual):
        def forward(self,*values):
            if self.context['frame']<hold:return torch.zeros_like(values[0])
            return super().forward(*values)
    class Chain(BatchedBControlChain):
        def __init__(self,*values,**kw):
            kw['mcc_config']=configured_mcc(replace(kw['mcc_config'],back_contact_x_limit_m=(1.,)*4),profile)
            super().__init__(*values,**kw);self.runtime.policy.diffusion.num_inference_steps=20
            graph=CUDAGraphHandFK(self.mcc.fk,B,self.device,frame_cache=True);self.mcc.fk=graph;self.direction.fk=graph
            attach_force_profile(self,observer,profile)
            self.async_planner=AsyncDPPlanner(self,minimum_latency_frames=int(np.ceil(args.latency_ms/10)),buffer_frames=30,max_age_frames=20)
        def prime(self,*values,**kw):
            super().prime(*values,**kw);self.control_graphs=ControlGraphKernels(self)
            # Attach AFTER graph preparation: the installed-chunk observation
            # wrapper must run at every live installation, not only capture.
            if args.worker=='residual':self.residual_policy=Residual(self,agent,observer,bounds,cfg,evaluate=True)
        def initial_state_gate(self,frame,q):
            if frame!=hold:return torch.zeros(B,dtype=torch.bool,device=device)
            mask=arrays['strict'][hold-20:hold].float()
            valid=(mask.all(-1).float().mean(0)>=.95)&(mask.mean(0).amin(-1)>=.95)
            valid&=torch.isfinite(q).all(-1)&(q>=self.mcc.lower-1e-5).all(-1)&(q<=self.mcc.upper+1e-5).all(-1)
            return ~valid
        def set_training_context(self,**ctx):
            self.context=ctx
            if self.residual_policy is not None:self.residual_policy.context=ctx
        def step(self,**kw):
            ctx=self.context
            if not ctx['settling']:
                f=ctx['frame'];x=observer.latest;alive=ctx['active'];seen=alive|ctx['newly_stopped']
                strict=x['pad_found']&torch.isfinite(x['pad_points']).all(-1)&torch.isfinite(x['pad_force']).all(-1)&(x['pad_force'].norm(dim=-1)>=.05)
                arrays['strict'][f]=strict&alive[:,None];arrays['found'][f]=(x['pad_found']|x['connector_found'])&alive[:,None]
                arrays['seen'][f]=seen;arrays['force'][f]=torch.where(seen,observer.safety_peak_force_n,0)
                arrays['q'][f]=kw['q'];origins=ctx['palm_pose'][:,:3]-torch.as_tensor(data['palm_pose_object'][f,:,:3],device=device)
                arrays['points'][f]=x['pad_points']-origins[:,None]
                arrays['pad_force'][f]=x['pad_force'];arrays['connector_force'][f]=x['connector_force']
                if f%1000==0:print(f'[LONG] {args.worker} frame={f}/{T} active={int(alive.sum())}/{B}',flush=True)
            action,diag=super().step(**kw)
            if not ctx['settling']:
                arrays['dp'][f]=self.previous_nominal;arrays['pre_mcc'][f]=diag['nominal_q'];arrays['command'][f]=diag['command_q']
                if self.residual_policy is not None:arrays['offset'][f]=self.residual_policy.offset
            return action,diag
    deployment.MODEL=str(args.dp)
    meta=deployment.run(B,rows[0]['trajectory_id'],T,device,args.seed,episode_data=data,chain_factory=Chain,
        contact_observer=observer,per_env_safety=True,settle_steps=100,max_safe_force_n=100,stop_when_all_inactive=True)
    if meta['stop_reason'] is not None:raise RuntimeError(meta)
    values={k:v.cpu().numpy() for k,v in arrays.items()};oracle=MeshNormalOracle.from_config(load_object_config('ycb_mustard'))
    lanes=[summarize_lane(r,values,meta,i,hold,oracle) for i,r in enumerate(rows)]
    trace=args.output.with_suffix('.npz');np.savez_compressed(trace,**values)
    report=dict(arm=args.worker,lanes=lanes,deployment=meta,trace=str(trace),dp=str(args.dp),residual=str(args.residual),
        metric_definition=observer.definition(),mcc_profile=profile,residual_original_mcc_profile=cfg['mcc_profile'],
        actor_updates=0,initialization_frames=hold,task_frames=rows[0]['frames'],
        scope='Continuous live DP+MCC, optionally frozen diagnostic R0; no teacher q future, no mid-task resets or history clearing',
        coverage='Approximate10mm source-mesh triangle-centroid cells, area-weighted; not exact tactile patch area',
        normals='Accepted source-mesh normal simulator retained; mesh is not used for finger IK/replanning')
    write(args.output,report);print('[LONG COMPLETE] '+json.dumps([(r['name'],r['status'],r['strict_healthy_four_rate'],r['found_four_rate']) for r in lanes]),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DATA/'long_palm_deployment_20261012')
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--worker',choices=['zero','residual'])
    parser.add_argument('--ids',type=int,nargs='+');parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--dp',type=Path,default=DP);parser.add_argument('--residual',type=Path,default=RESIDUAL)
    parser.add_argument('--latency-ms',type=float,default=80.);parser.add_argument('--seed',type=int,default=20261018)
    args=parser.parse_args()
    if args.worker:
        manifest=json.load(open(args.output.parent/'manifest.json'));worker(args,manifest);return
    args.output.mkdir(parents=True,exist_ok=True);manifest=prepare(args.output)
    print('[LONG INVENTORY] '+json.dumps({k:manifest[k] for k in ['recorded_candidates','existing_palm_references','deployable_palm_references']}),flush=True)
    if args.prepare_only:return
    ids=[r['trajectory_id'] for r in manifest['rows'] if r['status']=='ready'];reports=[]
    for start in range(0,len(ids),args.batch):
        chunk=ids[start:start+args.batch]
        for arm in ['zero','residual']:
            name=f'batch_{start//args.batch:02d}_{arm}';out=args.output/(name+'.json')
            if not out.exists():
                command=[sys.executable,__file__,'--worker',arm,'--ids',*[str(i) for i in chunk],
                    '--dp',str(args.dp),'--residual',str(args.residual),'--latency-ms',str(args.latency_ms),'--seed',str(args.seed+start),'--output',str(out)]
                print(f'[LONG RUN] {name}, {len(chunk)} lanes,6000 task frames',flush=True)
                with out.with_suffix('.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            reports.append(json.load(open(out)));write(args.output/'progress.json',dict(completed_batches=len(reports),remaining_batches=2*((len(ids)+args.batch-1)//args.batch)-len(reports)))
    summary=dict(manifest=manifest,arms={})
    for arm in ['zero','residual']:
        rows=[r for report in reports if report['arm']==arm for r in report['lanes']];legal=[r for r in rows if r['initial_physically_valid']]
        summary['arms'][arm]=dict(lanes=rows,initial_valid=len(legal),initial_failed=len(rows)-len(legal),
            reference_completed=sum(r['status']=='reference_completed' for r in legal),force_stops=sum(r['status']=='force_stop' for r in legal),
            strict_mean=float(np.mean([r['strict_healthy_four_rate'] for r in legal])) if legal else None,
            found_mean=float(np.mean([r['found_four_rate'] for r in legal])) if legal else None,
            reference_arc_mean=float(np.mean([r['reference_arc_fraction'] for r in legal])) if legal else None)
    write(args.output/'summary.json',summary);print(json.dumps({a:{k:v for k,v in r.items() if k!='lanes'} for a,r in summary['arms'].items()},indent=2),flush=True)


if __name__=='__main__':main()
