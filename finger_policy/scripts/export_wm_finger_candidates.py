"""Causal per-finger measured-label shards from audited WM trajectories."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
import h5py,numpy as np,torch
from single_finger_dp import project_history,robot_limits,HISTORY,HORIZON,STRIDE
from reconstruct_contact_wm_dp_observation import reconstruct

def run(manifest,output):
 if output.exists():raise FileExistsError(output)
 output.mkdir(parents=True);groups=defaultdict(list)
 for r in json.loads(manifest.read_text())['rows']:
  if r['candidate']:groups[r['folder']].append(r)
 limits=robot_limits();records=[]
 for folder,rows in groups.items():
  folder=Path(folder);digest=hashlib.sha256((folder/'rollout.h5').read_bytes()).hexdigest()
  assert all(r['rollout_sha256']==digest for r in rows),'Stale source audit'
  with h5py.File(folder/'rollout.h5') as h:
   active=h['raw_100hz/active_finger'][:].astype(bool);q=h['raw_100hz/q'][:];meta=json.loads(h.attrs['metadata'])
  states,parity=reconstruct(folder)
  for r in rows:
   lane,f=r['lane'],r['finger'];record=dict(r,source_sha256=digest);reasons=[]
   if not active[:,lane,f].all():reasons.append('finger_not_always_active')
   x=q[:len(states),lane,4*f:4*f+4];margin=np.minimum(x-limits[f,:,0],limits[f,:,1]-x)
   if margin.min()<0:reasons.append('measured_joint_limit_violation')
   split=meta.get('split')
   if split not in ['train','validation','test']:reasons.append('unknown_split')
   if parity.get('online_comparison_pass') is False:reasons.append('online_observation_parity_failed')
   record.update(min_joint_margin_rad=float(margin.min()),rejection_reasons=reasons,training_admissible=not reasons,split='val' if split=='validation' else split,object_parent=meta['object_id'])
   if not reasons:
    sampled=torch.from_numpy(states[::STRIDE,lane]);obs=project_history(sampled,[f],limits).numpy()[0];target=x[::STRIDE]
    assert obs.shape==(len(target),45) and np.array_equal(obs[:,:4],target)
    roots=np.arange(HISTORY-1,len(target)-HORIZON,dtype=np.int64)
    if len(roots)==0:raise ValueError('No full causal window')
    sid=hashlib.sha256((digest+f':{lane}:{f}').encode()).hexdigest()[:20];path=output/(sid+'.npz')
    np.savez_compressed(path,observation=obs,q=target,roots=roots,finger=f,joint_limits=limits[f])
    record.update(shard=str(path.resolve()),sampled_frames=len(target),windows=len(roots),family='thumb' if f==3 else 'long')
   records.append(record)
  report=dict(records=records,completed_folders=len({r['folder'] for r in records}),requested_folders=len(groups),complete=False,label='future measured q_i, same current-q anchor; nominal targets never labels')
  (output/'manifest.json').write_text(json.dumps(report,indent=2));print(folder.name,'processed',len(records),flush=True)
 report['complete']=True;(output/'manifest.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);run(a.manifest,a.output)
