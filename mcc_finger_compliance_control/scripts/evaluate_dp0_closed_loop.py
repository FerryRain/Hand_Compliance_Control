"""Closed-loop evaluation of DP0 alone on its untouched test paths.

Actual split-tip physics, pad-only
control feedback, free palm/substep recenter, scheduler/MCC and100N stop freeze.
Reports full-horizon and post-bootstrap strict healthy/found contact together.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import h5py,numpy as np,torch
import deploy_b_gpu_serial_parity as deployment
from deploy_dp_inverse import _episode,load_episode
from batched_b_control_chain import BatchedBControlChain
from true_fingertip_contact import ContactSlotObserver
from dp_motion_features import Q_VELOCITY_SLICE
from kinematic_baseline import absolute_kinematic_chunk

ROOT=Path(__file__).resolve().parents[2];JOB=ROOT/'mcc_finger_compliance_control/data/dp0_true_tip_1500'
MODELS=ROOT/'mcc_finger_compliance_control/data/models'
DP0_MODEL=MODELS/'DP0_true_tip_1500/best.pt'


def scheduler_scale():
 # Use the new policy's native deployment settings; no old-B dependency.
 checkpoint=torch.load(DP0_MODEL,map_location='cpu',weights_only=False)
 return np.asarray(checkpoint['normalization']['state_std'])[:16].copy()


def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def pack(ids,frames):
 rows=[load_episode(JOB/'DP0_inverted.h5',ep,include_teacher_tactile=False) for ep in ids]
 data={k:np.stack([r[k][:frames] for r in rows],axis=1) for k in ('q_hand','palm_pose_object','palm_twist_object')}
 with h5py.File(JOB/'DP0_commanded_palm.h5') as f:
  commands=[_episode(f,ep,'planner_palm_delta_pose_palm')[:frames] for ep in ids]
  for ep,row in zip(ids,rows):
   if not np.array_equal(_episode(f,ep,'q_hand')[:frames],row['q_hand'][:frames]):raise ValueError('reference/training source mismatch')
 data['planner_palm_delta_pose_palm_commanded']=np.stack(commands,axis=2)
 return data


class Metrics:
 def __init__(self,observer,batch,frames,device):
  self.observer=observer;self.frames=frames;self.batch=batch;self.device=device
  self.mask=torch.zeros((frames,batch,4),dtype=torch.bool,device=device)
  self.found=torch.zeros_like(self.mask);self.force=torch.zeros((frames,batch),device=device)
  self.q=torch.zeros((frames,batch,16),device=device);self.observed=torch.zeros((frames,batch),dtype=torch.bool,device=device)
  self.nominal=torch.zeros_like(self.q);self.command=torch.zeros_like(self.q)
  self.drop=torch.zeros((frames,batch),device=device,dtype=torch.long)
  self.initial=None
 def record(self,ctx,chain):
  if ctx['settling']:return
  frame=ctx['frame'];x=self.observer.latest
  alive=ctx['active'];seen=alive|ctx['newly_stopped']
  valid=x['pad_found']&torch.isfinite(x['pad_points']).all(-1)&torch.isfinite(x['pad_force']).all(-1)&(x['pad_force'].norm(dim=-1)>=.05)
  self.mask[frame]=valid&alive[:,None]
  self.found[frame]=(x['pad_found']|x['connector_found'])&alive[:,None]
  self.force[frame]=torch.where(seen,self.observer.safety_peak_force_n,0)
  if frame==0:self.initial=dict(q=ctx['q'].cpu().tolist(),pad_force=x['pad_force'].cpu().tolist(),palm_pose=ctx['palm_pose'].cpu().tolist())
  self.q[frame]=ctx['q'];self.observed[frame]=seen
 def report(self,ids,meta,reference,noise):
  mask=self.mask.cpu().numpy();found=self.found.cpu().numpy();force=self.force.cpu().numpy()
  q=self.q.cpu().numpy();seen=self.observed.cpu().numpy();teacher=reference['q_hand']
  result=[]
  for lane,ep in enumerate(ids):
   row=dict(episode_id=int(ep),safety_stop_frame=meta['per_env_stop_frame'][lane],peak_force_n=float(force[:,lane].max()))
   for start,label in [(0,'full'),(76,'post_bootstrap')]:
    if start>=self.frames:continue
    strict=mask[start:,lane];body=found[start:,lane];all4=strict.all(-1)
    longest=run=0
    for ok in all4:run=0 if ok else run+1;longest=max(longest,run)
    frame_seen=seen[start:,lane];fn=force[start:,lane][frame_seen]
    drift=np.abs(q[start:,lane]-teacher[start:,lane]).mean(-1)
    row[label]=dict(strict_healthy_four_rate=float(all4.mean()),strict_healthy_per_finger=strict.mean(0).tolist(),
      found_four_rate=float(body.all(-1).mean()),found_per_finger=body.mean(0).tolist(),
      longest_four_loss_frames=longest,force_p95_n=float(np.quantile(fn,.95)) if len(fn) else None,
      teacher_q_mae_rad=float(drift[frame_seen].mean()) if frame_seen.any() else None,
      observed_frames=int(frame_seen.sum()),unexecuted_tail_frames=int((~frame_seen).sum()))
    row[label]['success90']=bool(row['safety_stop_frame']<0 and all4.mean()>=.9 and strict.mean(0).min()>=.9)
    row[label]['success98']=bool(row['safety_stop_frame']<0 and all4.mean()>=.98 and strict.mean(0).min()>=.98 and longest<=10)
   result.append(row)
  return dict(lanes=result,deployment=meta,initial=self.initial,noise_schedule=noise,metric_definition=self.observer.definition(),
    scoring='All requested frames, including stopped tails as failed contact; post_bootstrap begins at76; force/q diagnostics use observed frames')


def evaluate(ids,frames,seed,device,trace_path=None,contact_gate='legacy',chunk_source='dp',diagnostics=False,inference_steps=None,mcc_profile=None,scheduler_scale_override=None):
 observer=ContactSlotObserver(controller_scope='pad',capture_collision=bool(mcc_profile and mcc_profile.get('projection')))
 metric=Metrics(observer,len(ids),frames,device);data=pack(ids,frames)
 frozen_scale=scheduler_scale() if scheduler_scale_override is None else np.asarray(scheduler_scale_override,dtype=np.float64)
 old=deployment.MODEL
 deployment.MODEL=str(DP0_MODEL)
 noise=[]
 extra={}
 if diagnostics:
  for name,width,dtype in [('pad_force',12,torch.float32),('connector_force',12,torch.float32),
   ('pad_points',12,torch.float32),('palm_pose',7,torch.float32),('pad_found',4,torch.bool),
   ('connector_found',4,torch.bool),('mcc_valid',4,torch.bool),('mcc_safety',4,torch.bool),
   ('mcc_filtered_force',4,torch.float32),('mcc_normal_offset',4,torch.float32),
   ('mcc_phase',4,torch.long),('recovery_offset',16,torch.float32),
   ('control_normal',12,torch.float32),('raw_inward',12,torch.float32),('planner_command',6,torch.float32)]:
   extra[name]=torch.zeros((frames,len(ids),width),device=device,dtype=dtype)
 class MeasuredChain(BatchedBControlChain):
  def __init__(self,*args,**kwargs):
   if contact_gate in ('whole_tip','whole_tip_guard'):
    # Sensor geom IDs already exclude the connector. Keep the anchor distance
    # sanity check, but do not reject legitimate side/back fingertip contacts.
    from dataclasses import replace
    kwargs['mcc_config']=replace(kwargs['mcc_config'],back_contact_x_limit_m=(1.,1.,1.,1.))
   elif contact_gate!='legacy':raise ValueError(contact_gate)
   if mcc_profile:
    from mcc_force_profiles import configured_mcc
    kwargs['mcc_config']=configured_mcc(kwargs['mcc_config'],mcc_profile)
   super().__init__(*args,**kwargs)
   if mcc_profile:
    from mcc_force_profiles import attach_force_profile
    attach_force_profile(self,observer,mcc_profile)
   if inference_steps is not None:self.runtime.policy.diffusion.num_inference_steps=inference_steps
   if contact_gate=='whole_tip_guard':
    from true_tip_collision_guard import TrueTipCollisionGuard
    self.collision_guard=TrueTipCollisionGuard(self.mcc)
   self.scheduler.scale=torch.as_tensor(frozen_scale,device=self.device,dtype=torch.float64).clamp_min(.02)
   original=self.runtime.policy.diffusion.conditional_sample
   def sample(*args,**kwargs):
    generator=kwargs['generator'];before=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
    result=original(*args,**kwargs)
    after=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
    noise.append(dict(frame=self.control_frame,before=before,after=after))
    return result
   self.runtime.policy.diffusion.conditional_sample=sample
   if chunk_source=='kinematic':
    def provider(frame,q,history):
     return absolute_kinematic_chunk(q,history[:,-1,Q_VELOCITY_SLICE],self.runtime.pred_horizon,
       self.runtime.config.action_waypoint_dt,self.runtime.config.kinematic_velocity_clip_rad_s)
    self.chunk_provider=provider
   elif chunk_source!='dp':
    field={'teacher_q_hand':'q_hand','teacher_q_prior':'q_prior'}[chunk_source]
    with h5py.File(JOB/'DP0_inverted.h5') as f:
     teacher=torch.as_tensor(np.stack([_episode(f,ep,field)[:frames] for ep in ids],axis=1),device=device)
    def provider(frame,q,history):
     index=(frame+torch.arange(1,self.runtime.pred_horizon+1,device=q.device)*self.runtime.stride).clamp_max(len(teacher)-1)
     return teacher[index].permute(1,0,2).contiguous()
    self.chunk_provider=provider
   if diagnostics:
    original_update=self.mcc.update
    def update(**values):
     self.control_normal=values['surface_normals_world']
     return original_update(**values)
    self.mcc.update=update
  def set_training_context(self,**ctx):
   self.context=ctx
  def step(self,**kwargs):
   self.context['q']=kwargs['q'];metric.record(self.context,self)
   action,diagnostic=super().step(**kwargs)
   if contact_gate=='whole_tip_guard' and not self.context['settling']:
    command=self.collision_guard(kwargs['q'],diagnostic['command_q'],kwargs['palm_pose_world'],
      kwargs['palm_twist_world'],observer.latest)
    diagnostic['command_q']=command
    action=((command-kwargs['q'])/.08).clamp(-2,2)
   if not self.context['settling']:
    frame=self.context['frame']
    metric.nominal[frame]=diagnostic['nominal_q'];metric.command[frame]=diagnostic['command_q']
    metric.drop[frame]=self.scheduler.last_drop_index
    if diagnostics:
     for name in ['pad_force','connector_force','pad_points','pad_found','connector_found']:
      extra[name][frame]=observer.latest[name].reshape(len(ids),-1)
     for name,value in [('palm_pose',kwargs['palm_pose_world']),('mcc_valid',diagnostic['pad_contact_valid']),
      ('mcc_safety',diagnostic['collision_safety_active']),('mcc_filtered_force',self.mcc.force_filtered),
      ('mcc_normal_offset',diagnostic['normal_offset']),('mcc_phase',diagnostic['contact_phase']),
      ('recovery_offset',self.runtime_recovery_offset),('control_normal',self.control_normal),
      ('raw_inward',kwargs['contact_normals_inward_world']),('planner_command',kwargs['planner_command_palm'])]:
      extra[name][frame]=value.reshape(len(ids),-1)
   return action,diagnostic
 try:
  meta=deployment.run(len(ids),ids[0],frames,device,seed,settle_steps=100,
    max_safe_force_n=100,per_env_safety=True,episode_data=data,chain_factory=MeasuredChain,contact_observer=observer)
  if meta['completed_frames']!=frames or meta['stop_reason'] is not None:raise RuntimeError(f'execution chain error: {meta}')
  report=metric.report(ids,meta,data,noise)
  report['scheduler_scale']=frozen_scale.tolist()
  report['contact_gate']=contact_gate;report['chunk_source']=chunk_source
  report['inference_steps_override']=inference_steps
  if mcc_profile:report['mcc_profile']=mcc_profile
  if chunk_source=='kinematic':
   if noise:raise AssertionError('DP was sampled during kinematic-only deployment')
   report['baseline_definition']='live q + future time * live causal50ms q velocity, clipped at1rad/s; no diffusion sampling; existing bootstrap and execution chain'
  report['input_sha256']=hashlib.sha256(b''.join(np.ascontiguousarray(data[k]).tobytes() for k in sorted(data))).hexdigest()
  if trace_path is not None:
   trace_path.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(trace_path,episode_ids=ids,q_live=metric.q.cpu().numpy(),
    q_teacher=data['q_hand'],q_nominal=metric.nominal.cpu().numpy(),q_command=metric.command.cpu().numpy(),
    strict=metric.mask.cpu().numpy(),found=metric.found.cpu().numpy(),
    force=metric.force.cpu().numpy(),observed=metric.observed.cpu().numpy(),drop_index=metric.drop.cpu().numpy(),
    **{k:v.cpu().numpy() for k,v in extra.items()})
   report['trace_path']=str(trace_path)
  return report
 finally:deployment.MODEL=old


def summarize(output,plan):
 arm=plan['arms'][0]
 lanes=[]
 for batch in plan['batches']:
  lanes.extend(json.loads((output/f"batch_{batch['index']:02d}_{arm}.json").read_text())['lanes'])
 aggregates={}
 for scope in ['full','post_bootstrap']:
  aggregates[scope]=dict(strict_healthy_four_rate=float(np.mean([r[scope]['strict_healthy_four_rate'] for r in lanes])),
   found_four_rate=float(np.mean([r[scope]['found_four_rate'] for r in lanes])),
   strict_healthy_per_finger=np.mean([r[scope]['strict_healthy_per_finger'] for r in lanes],axis=0).tolist(),
   found_per_finger=np.mean([r[scope]['found_per_finger'] for r in lanes],axis=0).tolist(),
   success90_count=sum(r[scope]['success90'] for r in lanes),success98_count=sum(r[scope]['success98'] for r in lanes),
   safety_stops=sum(r['safety_stop_frame']>=0 for r in lanes),lane_rollouts=len(lanes))
  for metric_name in ['strict_healthy_four_rate','found_four_rate']:
   groups=[np.array([r[scope][metric_name] for r in lanes if r['episode_id']==ep]) for ep in plan['episodes']]
   rng=np.random.default_rng(20261010);samples=[]
   for _ in range(5000):
    chosen=rng.integers(0,len(groups),len(groups));samples.append(np.mean([rng.choice(groups[j],len(groups[j]),replace=True).mean() for j in chosen]))
   aggregates[scope][metric_name+'_bootstrap95']=np.quantile(samples,[.025,.975]).tolist()
 episodes=[]
 for ep in plan['episodes']:
  rows=[r for r in lanes if r['episode_id']==ep]
  episodes.append(dict(episode_id=ep,
   strict_healthy_four_rate=float(np.mean([r['post_bootstrap']['strict_healthy_four_rate'] for r in rows])),
   found_four_rate=float(np.mean([r['post_bootstrap']['found_four_rate'] for r in rows])),
   success90_count=sum(r['post_bootstrap']['success90'] for r in rows),
   safety_stops=sum(r['safety_stop_frame']>=0 for r in rows),peak_force_n=max(r['peak_force_n'] for r in rows)))
 result=dict(aggregate=aggregates,per_episode=episodes,lanes=lanes,
  primary_scope='post_bootstrap',threshold_goal=.90,finger_order=['index','middle','ring','thumb'],
  scope='DP0 alone; all untouched test episodes retained; no Oracle filtering or B221 replay',
  observation='Live q/contact history and preissued palm command; teacher q only initializes and bootstraps frames0..75',
  uncertainty='Hierarchical bootstrap: trajectory then diffusion branch; physical numerical variance also contributes',
  historical_comparison='Historical B221 used different geometry/horizon/test paths; only contextual reference, not paired evidence')
 if plan.get('chunk_source')=='kinematic':
  result['scope']='Constant-velocity baseline alone; identical held-out paths and execution chain; no diffusion inference'
  result['uncertainty']='Hierarchical bootstrap: trajectory then physical repeat; baseline has no diffusion randomness'
 write(output/'summary.json',result)
 with (output/'per_lane.csv').open('w',newline='') as f:
  fields=['episode_id','branch','safety_stop_frame','peak_force_n','strict_healthy_four_rate','found_four_rate','longest_four_loss_frames','teacher_q_mae_rad']
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for i,row in enumerate(lanes):
   writer.writerow(dict(branch=i%plan['branches'],**{k:row[k] for k in fields[0:1]+fields[2:4]},**{k:row['post_bootstrap'][k] for k in fields[4:]}))
 return result


def analyze(output):
 """CPU-only analysis of completed, immutable rollout traces."""
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 summary=json.loads((output/'summary.json').read_text());plan=json.loads((output/'plan.json').read_text())
 arm=plan['arms'][0]
 traces=[]
 for batch in plan['batches']:
  with np.load(output/f"batch_{batch['index']:02d}_{arm}_trace.npz") as f:
   traces.append({k:f[k] for k in f.files})
 values={k:np.concatenate([t[k] for t in traces],axis=1) for k in traces[0] if k!='episode_ids'}
 ids=np.concatenate([t['episode_ids'] for t in traces]);seen=values['observed'];start=76
 drift=np.abs(values['q_live']-values['q_teacher']).mean(-1)
 tracking=np.abs(values['q_live'][1:]-values['q_command'][:-1]).mean(-1)
 tracking_seen=seen[1:]&seen[:-1]
 split=json.loads((JOB/'DP0_split.json').read_text())
 source={r['episode_id']:r['related_plan_group_id'] for r in split['episodes']}
 families={}
 import re
 for row in summary['per_episode']:
  family=re.split(r'_c\d+',source[row['episode_id']])[0]
  families.setdefault(family,[]).append(row)
 family_report={k:dict(episodes=len(rows),strict_healthy_four_rate=float(np.mean([r['strict_healthy_four_rate'] for r in rows])),
  found_four_rate=float(np.mean([r['found_four_rate'] for r in rows])),safety_stops=sum(r['safety_stops'] for r in rows)) for k,rows in families.items()}
 source_rates=[]
 with h5py.File(JOB/'DP0_inverted.h5') as f:
  for ep in plan['episodes']:
   mask=_episode(f,ep,'true_fingertip_contact_valid')[:plan['frames']]>0
   body=_episode(f,ep,'fingertip_body_collision_found')[:plan['frames']]>0
   source_rates.append(dict(episode_id=ep,strict_healthy_four_rate=float(mask.all(-1).mean()),found_four_rate=float(body.all(-1).mean())))
 rows=summary['per_episode'];sc=summary['aggregate']['post_bootstrap'];lane_rows=summary['lanes']
 diagnostic=dict(family=family_report,source_recording=source_rates,
  source_recording_strict_mean=float(np.mean([r['strict_healthy_four_rate'] for r in source_rates])),
  episodes_mean_strict_ge90=sum(r['strict_healthy_four_rate']>=.9 for r in rows),
  episodes_all4_branches_success90=sum(r['success90_count']==plan['branches'] for r in rows),
  episodes_any_safety_stop=sum(r['safety_stops']>0 for r in rows),
  teacher_q_mae_observed_rad=float(drift[start:][seen[start:]].mean()),
  command_tracking_next_frame_mae_rad=float(tracking[start:][tracking_seen[start:]].mean()),
  tracking_semantics='q_live[t+1] vs MCC command[t]; before termination; no actuator clipping reconstruction',
  longest_four_loss_median=float(np.median([r['post_bootstrap']['longest_four_loss_frames'] for r in lane_rows])),
  worst_episodes=sorted(rows,key=lambda r:r['strict_healthy_four_rate'])[:8],
  best_episodes=sorted(rows,key=lambda r:r['strict_healthy_four_rate'],reverse=True)[:8])
 write(output/'diagnostic.json',diagnostic)
 fig,ax=plt.subplots(figsize=(13,5))
 ordered=sorted(rows,key=lambda r:r['strict_healthy_four_rate']);x=np.arange(len(ordered))
 for offset,key,label,color in [(-.18,'strict_healthy_four_rate','Strict true fingertip','#2667b5'),(.18,'found_four_rate','Found (tip + connector)','#ed9730')]:
  means=np.array([r[key] for r in ordered])*100
  groups=[np.array([r['post_bootstrap'][key] for r in lane_rows if r['episode_id']==ep['episode_id']])*100 for ep in ordered]
  ax.bar(x+offset,means,width=.36,label=label,color=color)
  ax.errorbar(x+offset,means,yerr=np.stack([means-np.array([g.min() for g in groups]),np.array([g.max() for g in groups])-means]),
   fmt='none',color='#444444',linewidth=.7,capsize=2)
 ax.axhline(90,linestyle='--',color='#ba2020',linewidth=1,label='90% target')
 ax.set_xticks(x,[str(r['episode_id']) for r in ordered]);ax.set_ylim(0,105)
 ax.set_xlabel('Held-out episode (sorted by strict rate)');ax.set_ylabel('Four-finger contact (%)')
 ax.set_title(f"DP0 closed loop: {len(rows)} paths x {plan['branches']} samples, frames 76-{plan['frames']-1}; stopped tails = loss")
 ax.legend(loc='upper left');ax.grid(axis='y',alpha=.2);fig.tight_layout()
 fig.savefig(output/'contact_by_episode.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(12,8));t=np.arange(plan['frames'])/100
 for key,label,color in [('strict','Strict true fingertip','#2667b5'),('found','Found','#ed9730')]:
  axes[0,0].plot(t,values[key].all(-1).mean(-1)*100,label=label,color=color)
 axes[0,0].axvline(.76,color='gray',linestyle=':');axes[0,0].set_ylim(0,105)
 axes[0,0].set_title('Across all requested rollouts');axes[0,0].set_ylabel('Four-finger contact (%)');axes[0,0].legend()
 names=['Index','Middle','Ring','Thumb'];x=np.arange(4)
 axes[0,1].bar(x-.18,np.array(sc['strict_healthy_per_finger'])*100,.36,label='Strict')
 axes[0,1].bar(x+.18,np.array(sc['found_per_finger'])*100,.36,label='Found')
 axes[0,1].set_xticks(x,names);axes[0,1].set_ylim(0,105);axes[0,1].set_title('Per-finger, after bootstrap');axes[0,1].legend()
 masked_force=np.where(seen,values['force'],np.nan)
 axes[1,0].plot(t,np.nanquantile(masked_force,.95,axis=1),label='P95 across observed lanes')
 axes[1,0].axhline(100,color='#ba2020',linestyle='--',label='Independent safety stop')
 axes[1,0].set_ylabel('Force (N)');axes[1,0].set_title('Stopped lanes excluded from force diagnostics');axes[1,0].legend()
 axes[1,1].plot(t,np.nanmean(np.where(seen,drift,np.nan),axis=1),label='Teacher q deviation (diagnostic)')
 axes[1,1].plot(t[1:],np.nanmean(np.where(tracking_seen,tracking,np.nan),axis=1),label='Next-frame command tracking')
 axes[1,1].set_ylabel('Joint MAE (rad)');axes[1,1].set_title('Observed frames only');axes[1,1].legend()
 for a in axes.flat:a.grid(alpha=.2)
 for a in axes[1]:a.set_xlabel('Time (s)')
 fig.tight_layout();fig.savefig(output/'closed_loop_diagnostics.png',dpi=180);plt.close(fig)
 return diagnostic


def main():
 global JOB, DP0_MODEL
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--dataset-dir',type=Path,default=JOB)
 p.add_argument('--checkpoint',type=Path,default=DP0_MODEL)
 p.add_argument('--chunk-source',choices=['dp','kinematic'],default='dp')
 p.add_argument('--episodes',type=int,nargs='+');p.add_argument('--branches',type=int,default=4)
 p.add_argument('--episodes-per-batch',type=int,default=8);p.add_argument('--frames',type=int,default=1500)
 p.add_argument('--seed',type=int,default=20261010);p.add_argument('--device',default='cuda:0')
 p.add_argument('--contact-gate',choices=['legacy','whole_tip','whole_tip_guard'],default='legacy')
 p.add_argument('--inference-steps',type=int,default=10);a=p.parse_args()
 JOB=a.dataset_dir.resolve();DP0_MODEL=a.checkpoint.resolve()
 if a.device!='cuda:0':raise ValueError('GPU evaluation only')
 split=json.loads((JOB/'DP0_split.json').read_text());test=[r['episode_id'] for r in split['episodes'] if r['split']=='test']
 eps=a.episodes or test
 if not set(eps)<=set(test):raise ValueError('DP0 held-out test paths only')
 a.output.mkdir(parents=True,exist_ok=True)
 batches=[dict(index=i,ids=[ep for ep in eps[start:start+a.episodes_per_batch] for _ in range(a.branches)],seed=a.seed+i)
  for i,start in enumerate(range(0,len(eps),a.episodes_per_batch))]
 model_stat=DP0_MODEL.stat()
 plan=dict(protocol_version=4,episodes=eps,branches=a.branches,frames=a.frames,batches=batches,
  DP0_checkpoint=str(DP0_MODEL),checkpoint_size=model_stat.st_size,checkpoint_mtime_ns=model_stat.st_mtime_ns,
  controller_scope='pad',MCC_preset='collection_matched_sensor',settle_steps=100,residual='off',physics='true_fingertip_contact_v1_split',
  scheduler_scale=scheduler_scale().tolist(),scheduler_scale_source='native DP0 state_std[:16]',arms=['DP0'],excluded_episodes=[],
  contact_gate=a.contact_gate,inference_steps=a.inference_steps)
 if a.chunk_source!='dp':
  plan['chunk_source']=a.chunk_source;plan['arms']=[a.chunk_source]
 planfile=a.output/'plan.json'
 if planfile.exists() and json.loads(planfile.read_text())!=plan:raise ValueError('resume plan changed')
 write(planfile,plan);started=time.monotonic()
 for batch in batches:
  for arm in plan['arms']:
   path=a.output/f"batch_{batch['index']:02d}_{arm}.json"
   if path.exists():continue
   print(f"[DP0-EVAL] batch{batch['index']} {arm}; lanes={len(batch['ids'])}; frames={a.frames}",flush=True)
   row=evaluate(batch['ids'],a.frames,batch['seed'],a.device,a.output/f"batch_{batch['index']:02d}_{arm}_trace.npz",
     contact_gate=a.contact_gate,inference_steps=a.inference_steps,chunk_source=a.chunk_source);write(path,row)
   print(f"[DP0-EVAL] {arm}: strict={np.mean([x['post_bootstrap']['strict_healthy_four_rate'] for x in row['lanes']]):.4f}, found={np.mean([x['post_bootstrap']['found_four_rate'] for x in row['lanes']]):.4f}, stops={sum(x['safety_stop_frame']>=0 for x in row['lanes'])}",flush=True)
   write(a.output/'progress.json',dict(batch=batch['index'],arm=arm,elapsed_s=time.monotonic()-started))
 result=summarize(a.output,plan)
 analyze(a.output)
 print(json.dumps(result['aggregate'],indent=2),flush=True)

if __name__=='__main__':main()
