"""Reconstruct causal96D DP inputs from measured WM records, without a policy.

This validates the conversion interface, not physics or teacher quality.
Actions for later imitation must be future measured q from admitted episodes.
"""
import argparse,json,hashlib
from pathlib import Path
import h5py,numpy as np,torch
from batched_b_control_chain import BatchedBObservation
from contact_wm_nominal import relative_dp_inputs

def reconstruct(folder):
 with h5py.File(folder/'rollout.h5') as h:
  r=h['raw_100hz'];meta=json.loads(h.attrs['metadata'])
  fields=['q','palm_pose','object_pose','pad_point_world','pad_normal_world','pad_physical_found']
  raw={k:torch.from_numpy(r[k][:]) for k in fields};frames=len(raw['q'])-1;batch=raw['q'].shape[1]
  saved=r['dp_input_state'][1:] if 'dp_input_state' in r else None
  actual_future=r['dp_future_object_command'][1:] if 'dp_future_object_command' in r else None
 # Published plans may be saved as float64; the online execution tensor uses q dtype.
 with np.load(folder/'object_path_plan.npz') as z:commands=torch.as_tensor(z['commanded'],dtype=raw['q'].dtype)
 if len(commands)<frames:raise ValueError('Incomplete published path')
 obs=BatchedBObservation(batch,'cpu');previous=None;states=[]
 for t in range(frames):
  future=commands[min(t+19,len(commands)-1)]
  if actual_future is not None and not np.array_equal(future.numpy(),actual_future[t]):raise ValueError('Future command timeline differs from online DP')
  virtual,point,normal,twist,command=relative_dp_inputs(raw['palm_pose'][t],raw['object_pose'][t],future,raw['pad_point_world'][t],raw['pad_normal_world'][t],previous)
  previous=virtual.clone()
  state,_=obs.push(q=raw['q'][t],palm_pose_world=virtual,contact_points_world=point,contact_normals_inward_world=normal,found=raw['pad_physical_found'][t],palm_twist_world=twist,planner_command_palm=command)
  states.append(state.numpy())
 states=np.stack(states)
 if not np.isfinite(states).all():raise ValueError('Nonfinite reconstructed observation')
 report=dict(frames=frames,lanes=batch,state_dim=96,causal_future_command_offset=19,state_index='raw[t] before action t; recorded online input lives at raw[t+1]',first_usable_history_frame=75,training_admissible=False,object=meta['object_id'],source_sha256=hashlib.sha256((folder/'rollout.h5').read_bytes()).hexdigest())
 if saved is not None:
  diff=np.abs(states-saved);report.update(online_comparison_max=float(diff.max()),online_comparison_p99=float(np.quantile(diff,.99)),online_comparison_pass=bool(diff.max()<1e-4))
 else:report['online_comparison_pass']=None
 return states,report

def main():
 p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 torch.set_num_threads(2);states,report=reconstruct(a.folder);a.output.mkdir(parents=True)
 np.savez_compressed(a.output/'states.npz',state=states)
 (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
