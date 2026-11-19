"""Bounded durable collection queue. Attempted, recorded and audited are distinct."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
import h5py

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--queue',type=Path,required=True);ap.add_argument('--limit',type=int);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    d=a.queue.resolve().parent;scripts=Path(__file__).resolve().parent;repo=scripts.parents[1]
    python=sys.executable;status=d/'status.json'
    if status.exists() and not a.resume:raise FileExistsError('Use explicit --resume after inspecting status')
    state=json.loads(status.read_text()) if status.exists() else dict(jobs=[],started_at=time.time())
    state.update(pid=os.getpid(),complete=False)
    def save():
        state['updated_at']=time.time();tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
    done={r['id'] for r in state['jobs']};count=0;queue=json.loads(a.queue.read_text())['jobs']
    for job in queue:
        if job['id'] in done:continue
        if a.limit is not None and count>=a.limit:break
        if shutil.disk_usage(d).free<15*2**30:state['error']='disk reserve below 15GiB';break
        p=json.loads(Path(job['protocol']).read_text());folder=Path(job['folder'])
        if folder.exists():state['error']='Existing unregistered output; inspect before resume: '+str(folder);break
        paths=[p['backend'],str(scripts),str(repo/'mcc_finger_compliance_control/scripts'),str(repo/'src')]
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(paths),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MUJOCO_GL='egl',MPLCONFIGDIR='/tmp/wm_mpl')
        item=dict(id=job['id'],folder=str(folder),phases=[],motion_lanes=0,motion_frames=0)
        for phase,cmd in [('collect',[python,'-u',str(scripts/job['collector']),'--protocol',job['protocol']]),('audit',[python,str(scripts/'audit_contact_wm_interior_batch.py'),'--folders',str(folder),'--output',str(d/(job['id']+'.audit.json'))])]:
            state.update(phase=phase,current=job['id']);save();print('START',job['id'],phase,flush=True)
            started=time.time()
            try:
                with (d/(job['id']+'.'+phase+'.log')).open('x') as f:
                    code=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode
            except subprocess.TimeoutExpired:code=124
            item['phases'].append(dict(phase=phase,returncode=code,elapsed_s=time.time()-started))
            print('END',job['id'],phase,code,flush=True)
            if code:break
        if (folder/'rollout.h5').exists():
            with h5py.File(folder/'rollout.h5') as h:
                item['motion_frames']=h['raw_100hz/q'].shape[0]-1;item['motion_lanes']=h['raw_100hz/q'].shape[1]
                item['terminal_reason']=json.loads(h.attrs['metadata']).get('terminal_reason')
        item['full_motion']=item['motion_frames']==job['expected_frames']
        item['audited']=any(x['phase']=='audit' and x['returncode']==0 for x in item['phases'])
        state['jobs'].append(item);count+=1;save()
    state['complete']=len(state['jobs'])==len(queue)
    state['phase']='queue_finished' if state['complete'] else 'paused_after_limit_or_error'
    state['recorded_lanes']=sum(x['motion_lanes'] for x in state['jobs'])
    state['full_length_lanes']=sum(x['motion_lanes'] for x in state['jobs'] if x['full_motion'])
    save();print(json.dumps({k:state[k] for k in ['phase','complete','recorded_lanes','full_length_lanes']}),flush=True)

if __name__=='__main__':main()
