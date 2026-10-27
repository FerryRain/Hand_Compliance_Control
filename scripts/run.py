"""Portable entrypoints for the Mustard222 release. Run from any directory."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
RUNTIME=ROOT/'runtime'
S=RUNTIME/'mcc_finger_compliance_control/scripts';sys.path[:0]=[str(S),str(RUNTIME/'src'),str(RUNTIME)]
os.environ['PYTHONPATH']=os.pathsep.join([str(RUNTIME/'src'),str(RUNTIME),str(S),os.environ.get('PYTHONPATH','')])
DATA=ROOT/'data/mustard222'
META=json.loads((ROOT/'configs/release.json').read_text())
def resolved_split():
 payload=json.loads((ROOT/'configs/episode_split.json').read_text())
 payload['dp_h5']=str(DATA/'teachers_commanded_palm.h5')
 for row in payload['episodes']:row['source']=str(ROOT/row['source'])
 path=ROOT/'outputs/configs/episode_split.json';path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(payload,indent=2));return path

def call(script,*args):
 subprocess.run([sys.executable,'-u',str(S/script),*map(str,args)],cwd=ROOT,check=True)
def request(a):
 rows=json.loads((ROOT/'configs/source_manifest.json').read_text())['episodes'];row=next(x for x in rows if x['episode_id']==a.episode)
 template=ROOT/'configs/deploy.json'
 d=json.loads(template.read_text());d.update(rows=[dict(row,source=str(ROOT/row['source']),branch=0)],dataset=str(DATA),checkpoint=str(ROOT/META['checkpoints'][a.model]),arm=a.model,chunk_source=a.source,output=str(ROOT/f'outputs/{a.mode}_{a.model}_ep{a.episode}/result.json'),persist_trace=True)
 d.pop('arm_feedforward',None)
 if a.frames:d['rows'][0]['frames']=min(a.frames,row['frames']);d['reference_prefix_only']=True
 out=Path(d['output']).parent;out.mkdir(parents=True,exist_ok=True);p=out/'request.json';p.write_text(json.dumps(d,indent=2));return p

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','train','collect','prepare','request','deploy','view']);p.add_argument('--model',choices=list(META['checkpoints']),default='flat24k');p.add_argument('--episode',type=int,default=48);p.add_argument('--frames',type=int);p.add_argument('--steps',type=int,default=25000);p.add_argument('--source',choices=['dp','teacher'],default='dp');p.add_argument('--tube',action='store_true');p.add_argument('--device',default='cuda:0');a=p.parse_args()
 os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
 if a.mode=='verify':
  import h5py,numpy as np,torch
  from deploy_dp_inverse import DPRuntime
  from train_dp import load_fixed_episode_split
  rows=json.loads((ROOT/'configs/source_manifest.json').read_text())['episodes']
  for row in rows:
   f=ROOT/row['source'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256'],str(f)
  load_fixed_episode_split(resolved_split(),DATA/'teachers_commanded_palm.h5',{r['episode_id'] for r in rows})
  for name,path in META['checkpoints'].items():
   runtime=DPRuntime(ROOT/path,torch.device('cpu'),10,42)
   y=runtime.predict(np.repeat(runtime.state_mean[None,:],16,axis=0).astype(np.float32))
   assert y.shape==(16,16) and np.isfinite(y).all();print(name,'checkpoint load / 16x16 inference PASS')
   del runtime
  print('222 source checksums, frozen split, flat CPU inference path: PASS')
 elif a.mode=='train':
  args=['--file',DATA/'teachers_commanded_palm.h5','--episode-split-manifest',resolved_split(),'--output',ROOT/f'outputs/train_{a.model}','--device',a.device,'--steps',a.steps,'--batch-size',64,'--lr',.0003,'--stride',5,'--obs-horizon',16,'--pred-horizon',16,'--input-profile','B2','--action-representation','anchored_delta_q','--diffusion-steps',100,'--inference-steps',10,'--noise-scheduler','DDPM','--down-dims',256,512,1024,'--fk-loss-weight',0,'--num-workers',4,'--seed',42,'--save-every',10000,'--eval-every',1000,'--eval-samples',64,'--conditioning-architecture','flat']
  call('train_dp.py',*args)
 elif a.mode=='collect':
  from run_variable_length_collection import make_command,execution_contract,rejection_reasons
  from run_leap_hardware_evaluation import analyze
  from plot_leap_hardware_evaluation import motion_statistics
  r=next((x for x in json.loads((ROOT/'configs/collection_references.json').read_text()) if x['episode_id']==a.episode),None)
  if r is None:raise SystemExit('This episode has no preserved planning reference; choose an ID in configs/collection_references.json')
  out=ROOT/f'outputs/collect_ep{a.episode}{"_tube" if a.tube else "_clean"}'
  if (out/'summary.json').exists():raise SystemExit('Collection already exists; move its output directory before recollecting')
  out.mkdir(parents=True,exist_ok=True)
  job=dict(reference=str(ROOT/r['reference']),output=str(out),frames=r['frames'],requested_frames=r['frames'],seed=20261027,tube=a.tube,candidate_id=a.episode,variant='accepted_10N_handover')
  subprocess.run(make_command(job),cwd=ROOT,check=True)
  result=analyze(job);result.update(job)
  if result.get('status')=='completed':
   result['offline_motion_statistics']=motion_statistics(out,result);result['peak_all_parts_n']=result['offline_motion_statistics']['peak_all_parts_n']
  result['execution_contract']=execution_contract(out,result);result['rejection_reasons']=rejection_reasons(result);result['selected']=not result['rejection_reasons']
  (out/'acceptance.json').write_text(json.dumps(result,indent=2));print('Selected:',result['selected'],result['rejection_reasons'])
 elif a.mode=='prepare':
  folder=ROOT/'outputs/reexport_inputs';folder.mkdir(parents=True,exist_ok=True)
  manifest=json.loads((ROOT/'configs/collection_manifest.json').read_text())
  for row in manifest['rows']:row['selected_file']=str(ROOT/row['selected_file'])
  (folder/'collection_manifest.json').write_text(json.dumps(manifest,indent=2))
  call('prepare_stratified_dp.py','--collection',folder,'--output',ROOT/'outputs/reexport222')
 else:
  req=request(a)
  if a.mode=='request':print(req)
  elif a.mode=='view':call('visualize_regional_deployment.py','--request',req)
  elif a.mode=='deploy':call('evaluate_regional_dp.py','--worker',req)
if __name__=='__main__':main()
