"""Tune MCC on development episodes, freeze it, then train a fresh R0.

No DP representation changes or DPPO updates are made in this initial stage.
Test episodes are diagnostic only; future final testing needs new planner paths.
"""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import evaluate_dp0_closed_loop as evaluation
from mcc_force_profiles import PROFILES

S=Path(__file__).resolve().parent;ROOT=S.parent.parent;DATA=S.parent/'data'
JOB=DATA/'mcc_residual_iteration_1500'
RL=DATA/'RL0/ppo_DP0_mcc_retrained_20261011'


def write(path,value):evaluation.write(path,value)


def status(phase,**fields):
    write(JOB/'status.json',dict(phase=phase,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields))
    print('[MCC-R0]',phase,fields,flush=True)


def aggregate(report):
    lanes=report['lanes']
    return dict(strict=float(np.mean([r['post_bootstrap']['strict_healthy_four_rate'] for r in lanes])),
        found=float(np.mean([r['post_bootstrap']['found_four_rate'] for r in lanes])),
        stops=sum(r['safety_stop_frame']>=0 for r in lanes),rollouts=len(lanes),
        force_p95=float(np.mean([r['post_bootstrap']['force_p95_n'] for r in lanes])),
        success90=sum(r['post_bootstrap']['success90'] for r in lanes))


def gate(base,new,require_stop_reduction):
    a,b=aggregate(base),aggregate(new);health=[]
    for ep in sorted({r['episode_id'] for r in base['lanes']}):
        x=[r for r in base['lanes'] if r['episode_id']==ep];y=[r for r in new['lanes'] if r['episode_id']==ep]
        healthy=all(r['safety_stop_frame']<0 and r['post_bootstrap']['strict_healthy_four_rate']>=.98 for r in x)
        change=np.mean([r['post_bootstrap']['strict_healthy_four_rate'] for r in y])-np.mean([r['post_bootstrap']['strict_healthy_four_rate'] for r in x])
        if healthy and (change<-.02 or any(r['safety_stop_frame']>=0 for r in y)):health.append(ep)
    safe=b['stops']<a['stops'] if require_stop_reduction else b['stops']<=a['stops']
    passed=safe and b['strict']>=a['strict']-.02 and b['found']>=a['found']-.02 and not health
    return dict(passed=bool(passed),baseline=a,candidate=b,healthy_regressions=health,
                tolerances='strict/found mean loss<=2pp; no new stop on previously healthy episodes; healthy loss<=2pp')


def run_profile(profile,stage,eps,branches=2):
    output=JOB/stage/profile['name'];output.mkdir(parents=True,exist_ok=True)
    plan=dict(episodes=eps,branches=branches,frames=1500,arms=['DP0'],mcc_profile=profile,
        batches=[dict(index=i,ids=[ep for ep in eps[k:k+8] for _ in range(branches)],seed=20261012+i)
                 for i,k in enumerate(range(0,len(eps),8))])
    p=output/'plan.json'
    if p.exists() and json.loads(p.read_text())!=plan:raise ValueError('MCC comparison protocol changed')
    write(p,plan)
    for batch in plan['batches']:
        p=output/f"batch_{batch['index']:02d}_DP0.json"
        if p.exists():continue
        status('evaluating_MCC',stage=stage,profile=profile['name'],batch=batch['index'],lanes=len(batch['ids']))
        # A fresh worker owns each CUDA graph. Process exit releases Warp pools
        # and closure cycles reliably; completed JSON batches remain resumable.
        request=p.with_suffix('.request.json')
        write(request,dict(ids=batch['ids'],seed=batch['seed'],profile=profile,
            output=str(p),trace=str(output/f"batch_{batch['index']:02d}_DP0_trace.npz")))
        log=output/f"batch_{batch['index']:02d}_worker.log"
        with log.open('a') as stream:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(request)],
                                  cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'MCC batch worker failed; inspect {log}')
    result=evaluation.summarize(output,plan)
    result['scope']=f'{stage}: development/diagnostic episodes; not locked final test; frozen DP0, only MCC changed'
    write(output/'summary.json',result)
    print('[MCC-PROFILE]',profile['name'],stage,aggregate(result),flush=True)
    return result


def main():
    JOB.mkdir(parents=True,exist_ok=True)
    lock=(JOB/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if not json.loads((JOB/'controller_unit_gate.json').read_text())['passed']:raise RuntimeError('Controller unit gate failed')
        split=json.loads((evaluation.JOB/'DP0_split.json').read_text())
        dev=[r['episode_id'] for r in split['episodes'] if r['split']=='val']
        screen=[3,20,54,56,17,21]
        if not set(screen)<=set(dev):raise ValueError('MCC tuning must stay within development trajectories')
        protocol=dict(DP0=str(evaluation.DP0_MODEL),data=str(evaluation.JOB),DPPO_started=False,
            DP='96D B2, U-Net, measured-q anchored_delta_q, history16/future16, DDPM100 training/20 inference',
            scheduler='50ms waypoints/100ms replan; unchanged',residual='50ms decisions with existing rate/offset interface',
            independent_force_stop_n=100,metrics=['strict split whole true fingertip','found tip or connector'],
            current_phase=['MCC unit parity','development MCC sweep','full-development safety/health gate','fresh PPO residual'],
            future_phase=['freeze accepted R0; DPPO likelihood/execution parity and cost canary',
                          'freeze R0; DP0->DP1','freeze accepted DP1; R0->R1','matched-budget residual-only control and new planner test'],
            development=dev,screen=screen,old_test_role='Repeatedly examined diagnostic set; final generalization requires new legal planner trajectories',
            profile_guard='Two-pp contact/healthy tolerance declared before results; stop line cannot be raised',
            residual_budget=1000000,source_notes='Connector safety uses simulated collision force; hardware needs corresponding load measurement/estimate',
            references={'DPPO':'https://arxiv.org/html/2409.00588v3','REFINE-DP':'https://arxiv.org/html/2603.13707v2'})
        write(JOB/'iteration_protocol.json',protocol)
        baseline=run_profile(PROFILES['baseline'],'screen',screen)
        ranked=[];screens={}
        for name,p in PROFILES.items():
            if name=='baseline':continue
            result=run_profile(p,'screen',screen);g=gate(baseline,result,False);screens[name]=g
            if g['passed']:ranked.append((g['candidate']['stops'],-g['candidate']['strict'],g['candidate']['force_p95'],name))
        write(JOB/'screen_comparison.json',screens)
        full_base=run_profile(PROFILES['baseline'],'development',dev)
        selected=None;acceptances={}
        for _,_,_,name in sorted(ranked):
            result=run_profile(PROFILES[name],'development',dev);g=gate(full_base,result,True);acceptances[name]=g
            write(JOB/'development_gates.json',acceptances)
            if g['passed']:selected=PROFILES[name];break
        if selected is None:
            status('MCC_no_accepted_candidate',screen_results=str(JOB/'screen_comparison.json'),
                   note='No profile both reduces stops and preserves contact; do not silently train with a rejected controller')
            return
        profile_file=JOB/'selected_mcc_profile.json';write(profile_file,selected)
        write(JOB/'MCC_decision.json',dict(selected=selected,gate=acceptances[selected['name']],force_stop_n=100,
            next='Fresh residual; later DPPO freezes this new residual and the selected MCC'))
        command=[sys.executable,str(S/'train_rl0.py'),'--output',str(RL),'--contact-gate','whole_tip',
                 '--mcc-profile',str(profile_file),'--num-envs','64','--transitions','1000000',
                 '--seed','20261012','--inference-steps','20']
        write(JOB/'R0_training_command.json',dict(command=command))
        if not (RL/'summary.json').exists():
            log=JOB/'logs/train_R0.log';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as f:
                process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    status('training_fresh_R0',profile=selected['name'],child_pid=process.pid,log=str(log),output=str(RL));time.sleep(20)
            if process.returncode:raise RuntimeError('Fresh R0 training failed; inspect '+str(log))
        summary=json.loads((RL/'summary.json').read_text())
        status('evaluating_trained_R0',profile=selected['name'],R0_summary=summary)
        log=JOB/'logs/R0_acceptance.log'
        with log.open('a') as stream:
            result=subprocess.run([sys.executable,str(S/'evaluate_rl0_development.py'),
                '--experiment',str(RL)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Full-development residual acceptance failed to run; inspect {log}')
        acceptance=json.loads((RL/'full_development/acceptance.json').read_text())
        status('R0_accepted' if acceptance['accepted'] else 'R0_not_accepted',
               profile=selected['name'],R0_summary=summary,acceptance=acceptance,
               next='DPPO interface parity and cost canary with frozen learned R0' if acceptance['accepted'] else
                    'Diagnose residual; do not treat step0 or a harmful policy as learned R0')
    except BaseException as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc());raise


def worker(request):
    payload=json.loads(Path(request).read_text())
    report=evaluation.evaluate(payload['ids'],1500,payload['seed'],'cuda:0',
        Path(payload['trace']),contact_gate='whole_tip',inference_steps=20,
        diagnostics=True,mcc_profile=payload['profile'])
    write(Path(payload['output']),report)


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(sys.argv[2])
    else:main()
