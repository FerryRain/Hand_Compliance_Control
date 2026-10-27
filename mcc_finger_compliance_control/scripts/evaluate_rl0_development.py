"""Paired full-development acceptance of a fresh residual and its zero baseline.

Each CUDA batch runs in a separate process; the actor, MCC and DP stay frozen.
Previously inspected test paths are not used to select the residual.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import train_rl0 as training
from run_mcc_then_residual import aggregate, gate


def worker(request):
    x=json.loads(Path(request).read_text())
    state=torch.load(x['checkpoint'],map_location='cpu',weights_only=False)
    cfg,bounds=state['config'],state['bounds']
    if training.digest(training.DP0_MODEL)!=x['DP0_sha256']:
        raise ValueError('Frozen DP0 changed after residual training')
    agent=training.RL0PPO(cfg,bounds)
    agent.actor.load_state_dict(state['actor'])
    agent.value.load_state_dict(state['value'])
    report,_=training.rollout(agent,cfg,bounds,x['ids'],x['seed'],
                              zero=x['zero'],evaluate=True)
    training.write(Path(x['output']),report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args();experiment=a.experiment.resolve()
    manifest=json.loads((experiment/'manifest.json').read_text())
    summary=json.loads((experiment/'summary.json').read_text())
    eps=manifest['split']['dev'];output=experiment/'full_development'
    output.mkdir(exist_ok=True)
    # A step-0 best is still evaluated and explicitly rejected as learned R0.
    checkpoints={'zero':experiment/'initial.pt','best':experiment/'best.pt'}
    states={k:torch.load(v,map_location='cpu',weights_only=False) for k,v in checkpoints.items()}
    for state in states.values():
        if state['config']!=manifest['config'] or state['bounds']!=manifest['bounds']:
            raise ValueError('Frozen residual configuration/bounds changed')
        if not all(torch.isfinite(v).all() for group in ['actor','value'] for v in state[group].values()):
            raise ValueError('Nonfinite residual checkpoint')
    actor_change=max(float((states['best']['actor'][k]-v).abs().max())
                     for k,v in states['zero']['actor'].items())
    plan=dict(episodes=eps,branches=2,frames=1500,
              DP0_sha256=manifest['DP0_sha256'],config=manifest['config'],
              checkpoint_sha256={k:training.digest(v) for k,v in checkpoints.items()},
              batches=[dict(index=i,ids=[ep for ep in eps[k:k+8] for _ in range(2)],
                            seed=manifest['config']['seed']+5000+i)
                       for i,k in enumerate(range(0,len(eps),8))])
    target=output/'plan.json'
    if target.exists() and json.loads(target.read_text())!=plan:
        raise ValueError('Acceptance protocol/checkpoints changed')
    training.write(target,plan)
    reports={}
    for arm,checkpoint in checkpoints.items():
        reports[arm]=[]
        for b in plan['batches']:
            result=output/f"{arm}_batch_{b['index']:02d}.json"
            if not result.exists():
                request=result.with_suffix('.request.json')
                training.write(request,dict(checkpoint=str(checkpoint),ids=b['ids'],
                    seed=b['seed'],zero=arm=='zero',output=str(result),
                    DP0_sha256=manifest['DP0_sha256']))
                print('[R0-ACCEPT]',arm,b['index'],len(b['ids']),flush=True)
                with result.with_suffix('.log').open('a') as stream:
                    child=subprocess.run([sys.executable,str(Path(__file__).resolve()),
                        '--worker',str(request)],cwd=training.ROOT,
                        stdout=stream,stderr=subprocess.STDOUT)
                if child.returncode:raise RuntimeError(f'Acceptance worker failed: {result.with_suffix(".log")}')
            reports[arm].append(json.loads(result.read_text()))
    paired_noise=all(x['noise_schedule']==y['noise_schedule']
                    for x,y in zip(reports['zero'],reports['best']))
    if not paired_noise:raise RuntimeError('Paired diffusion sampling schedules differ')
    base=dict(lanes=[r for b in reports['zero'] for r in b['lanes']])
    best=dict(lanes=[r for b in reports['best'] for r in b['lanes']])
    verdict=gate(base,best,False)
    deltas={}
    groups=[]
    for ep in eps:
        x=[r for r in base['lanes'] if r['episode_id']==ep]
        y=[r for r in best['lanes'] if r['episode_id']==ep]
        d=np.array([[v['post_bootstrap'][metric]-u['post_bootstrap'][metric]
                     for metric in ['strict_healthy_four_rate','found_four_rate']]
                    for u,v in zip(x,y)])
        groups.append(d);deltas[ep]=d.mean(0).tolist()
    rng=np.random.default_rng(20261012);boot=[]
    for _ in range(5000):
        chosen=rng.integers(0,len(groups),len(groups))
        boot.append(np.mean([groups[j][rng.integers(0,len(groups[j]),len(groups[j]))].mean(0)
                             for j in chosen],0))
    learned=summary['best_training_transitions']>0 and actor_change>0
    improvement=aggregate(best)['strict']>aggregate(base)['strict']
    accepted=learned and verdict['passed'] and improvement
    training.write(output/'acceptance.json',dict(
        accepted=accepted,learned_checkpoint=learned,
        actor_parameter_max_change=actor_change,finite_parameters=True,
        best_training_transitions=summary['best_training_transitions'],gate=verdict,
        paired_diffusion_schedule_verified=paired_noise,
        paired_episode_deltas_strict_found=deltas,
        paired_bootstrap95_strict_found=np.quantile(boot,[.025,.975],axis=0).T.tolist(),
        interpretation='Development estimate, not a locked final test; a positive point estimate is not proof of statistical significance',
        next='Freeze this learned R0 for DPPO interface parity and cost canary' if accepted else
             'Do not start DPPO automatically; diagnose residual behavior or zero-best selection first'))
    print('[R0-ACCEPT]',json.dumps(dict(accepted=accepted,baseline=aggregate(base),best=aggregate(best))),flush=True)


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--worker':worker(sys.argv[2])
    else:main()
