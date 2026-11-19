"""One expanded evaluation job. Evaluation body copied unchanged from frozen queue."""
import argparse,contextlib,io,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from controlled_policy_ablation import save,sha
from audit_smooth_regrasp import audit
ROOT=Path(__file__).resolve().parents[2]
def run(script,args,log,timeout):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as s:
        subprocess.run([sys.executable,'-u',str(Path(__file__).with_name(script)),*map(str,args)],cwd=ROOT,
            env=dict(os.environ,MUJOCO_GL='egl',MPLCONFIGDIR='/tmp/mpl-expanded-policy'),stdout=s,stderr=subprocess.STDOUT,check=True,timeout=timeout)

def evaluate(out,p,group,seed,ckpt,name,canary=False,*,denoise_steps=10,record_latency=False):
    folder=out/('canary_deployment' if canary else 'evaluation')/f'{group}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    target=folder/(name+'.json');reportpath=folder/(name+'.comparison_audit.json')
    task=json.loads(Path(p['request_templates'][name]['request']).read_text());arm='flat' if group=='flat_common' else 'single'
    task.update(arm=f'{group}_seed{seed}_{name}',single_finger_policy=dict(mode='all',comparison_checkpoint=str(ckpt.resolve())),
                comparison_checkpoint_sha256=sha(ckpt),output=str(target.resolve()),matched_progress_noise=False)
    task.pop('finger_checkpoint_hashes',None)
    if denoise_steps != 10 or record_latency:
        task['single_finger_policy'].update(denoise_steps=denoise_steps,record_latency=record_latency)
    if canary:task['rows'][0]['frames']=200;task['palm_reset']['extra_frames']=500;task['reference_prefix_only']=True
    request=folder/(name+'.request.json')
    if reportpath.exists():
        if not target.is_file() or json.loads(request.read_text())!=task:
            raise ValueError('Saved evaluation does not match the requested run')
        cached=json.loads(reportpath.read_text());assert cached['checkpoint_sha256']==sha(ckpt)
        return cached
    if target.exists():
        assert json.loads(request.read_text())==task,'Saved request mismatch'
    else:
        save(request,task);run('evaluate_controlled_policy_ablation.py',[request],folder/(name+'.log'),14400)
    result=json.loads(target.read_text());info=result['single_finger_policy']
    assert info['sha256']==sha(ckpt) and info['arm']==arm and info['calls']>0
    assert info.get('denoise_steps', 10)==denoise_steps
    with contextlib.redirect_stdout(io.StringIO()):report=audit(target)
    with np.load(result['trace']) as z:
        n=result['lanes'][0]['frames'];assert n>100
        assert all(np.isfinite(z[k][:n]).all() for k in ['q','command','nominal'])
        cursor=z['reset_reference_frame'][:n,0];phase=z['reset_phase'][:n,0]
        moving=np.r_[False,np.diff(cursor)>0]&(phase==0)
        report['advancing_only']=dict(frames=int(moving.sum()),strict_four_rate=float(z['strict'][:n,0][moving].all(-1).mean()) if moving.any() else None,
            found_four_rate=float(z['found'][:n,0][moving].all(-1).mean()) if moving.any() else None)
        report['requested_reference_frames']=task['rows'][0]['frames'];report['last_reference_frame']=int(cursor[-1])
        report['reference_completion_fraction']=(int(cursor[-1])+1)/task['rows'][0]['frames'] if result['lanes'][0]['complete'] else int(cursor[-1])/max(task['rows'][0]['frames']-1,1)
        # Keep force segments and bidirectional range as independent outcomes.
        from surface_mcc_finger import FullHandMCCFingerConfig,FullHandMCCFingerController
        ctl=FullHandMCCFingerController(FullHandMCCFingerConfig(**result['source_config']))
        q=z['q'][:n,0];margin=np.minimum(q-ctl.lower,ctl.upper-q)
        def segments(mask):
            edges=np.diff(np.r_[False,mask,False].astype(int));a=np.where(edges==1)[0];b=np.where(edges==-1)[0]
            return dict(frames=int(mask.sum()),fraction=float(mask.mean()),segments=len(a),longest=int((b-a).max(initial=0)))
        report['joint_margin']=dict(minimum=float(margin.min()),p05=float(np.quantile(margin,.05)),
            per_joint_minimum=margin.min(0).tolist(),per_joint_lower_clearance_min=(q-ctl.lower).min(0).tolist(),per_joint_upper_clearance_min=(ctl.upper-q).min(0).tolist(),outside=segments((margin < -1e-6).any(-1)))
        report['force_detail']={}
        for field,label in [('force','object'),('self_force','self')]:
            force=z[field][:n,0];force=force.reshape(n,-1).max(-1)
            report['force_detail'][label]=dict(peak=float(force.max()),p95=float(np.quantile(force,.95)),over100=segments(force>100))
    report['checkpoint_sha256']=sha(ckpt);report['canary_prefix_only']=canary;save(reportpath,report);return report

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path);ap.add_argument('group');ap.add_argument('seed',type=int);ap.add_argument('reference')
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--canary', action='store_true')
    ap.add_argument('--denoise-steps', type=int, default=10)
    ap.add_argument('--record-latency', action='store_true')
    a=ap.parse_args()
    p=json.loads((a.output/'protocol.json').read_text())
    ckpt=a.checkpoint or a.output/'training'/f'{a.group}_seed{a.seed}/best.pt'
    evaluate(a.output,p,a.group,a.seed,ckpt,a.reference,canary=a.canary,
             denoise_steps=a.denoise_steps,record_latency=a.record_latency)
