"""Full-duration comparison at the completion time of the archived Active-2 Aude run."""
import argparse
import json
from pathlib import Path
import shutil
from time import sleep

import numpy as np
import torch

from .evaluate import audit, read_trace, reconstruct, reference
from .experiment import ACTIVE2, CONTROL_DT, ROOT, digest, run, save_json


METHODS=('history-dp','history-mcc','aude-mcc','aude-single-all-mcc')


def prepare(root):
    dest=root/'pear-full'
    dest.mkdir(parents=True,exist_ok=True)
    for name in ('hand-history.npz','protocol.json'):
        if not (dest/name).exists():
            shutil.copy2(root/name,dest/name)
    times={a:json.loads((ACTIVE2/a/'run.json').read_bytes())['seconds'] for a in METHODS[2:]}
    spec=dict(object='pear',start='active-02',arms=METHODS,seconds=min(times.values()),
              original_end_s=times,checkpoints_s=[240.,600.,min(times.values())],
              coverage_targets=[.7,.95],maximum_region_load_strictly_below_n=8.,
              registered_after_240s_results=True,rule='entire common interval ending at first archived completion',
              script_sha256=digest(Path(__file__)))
    if (dest/'long-protocol.json').exists() and json.loads((dest/'long-protocol.json').read_bytes())!=json.loads(json.dumps(spec)):
        raise ValueError('Full-duration protocol changed')
    save_json(dest/'long-protocol.json',spec)


def evaluate(root,arm,spec):
    dest=root/'pear-full'
    archive=arm.startswith('aude')
    folder=dest/('archive' if archive else 'runs')/'pear/active-02'/arm
    folder.mkdir(parents=True,exist_ok=True)
    source=ACTIVE2/arm if archive else folder
    if (folder/'full-score.json').exists():
        return json.loads((folder/'full-score.json').read_bytes())
    length=round(spec['seconds']/CONTROL_DT)
    tr=read_trace(source/'trace.npz',length)
    assert len(tr['qpos'])==length+1
    src_hash=digest(source/'trace.npz')
    if archive:
        archived=json.loads((ACTIVE2/'signed-results.json').read_bytes())
        assert src_hash==next(x['trace_sha256'] for x in archived['runs'] if x['arm']==arm)
    else:
        receipt=json.loads((folder/'run.json').read_bytes())
        assert receipt['trace_sha256']==src_hash
        assert digest(folder/'plans.jsonl')==receipt['plans_sha256']
        plans=[json.loads(l) for l in (folder/'plans.jsonl').read_text(encoding='utf-8').splitlines()]
        audit(tr,plans,True)
    if archive:
        matches=True
    else:
        previous=read_trace(root/'runs/pear/active-02'/arm/'trace.npz',24000)
        matches=all(np.array_equal(previous[k],tr[k][:len(previous[k])]) for k in ('qpos','palm_command','pad_valid'))
    load=tr['region_load_n'].max(1)
    valid=tr['pad_valid']&(load<8.)[:,None]
    ticks,fingers=np.nonzero(valid)
    points=tr['points_m'][ticks,fingers];normals=tr['normals_outward'][ticks,fingers]
    ref=reference('pear')
    curves,_=ref.area(points,normals,ticks,length)
    snaps=[]
    for t in spec['checkpoints_s']:
        k=round(t/CONTROL_DT)
        select=ticks<=k
        signed=reconstruct(points[select],normals[select],ref,folder/f'signed-{t:g}s.ply','signed')
        snaps.append(dict(seconds=t,coverage10=float(curves[0,k]),coverage25=float(curves[1,k]),signed=signed))
    hits={}
    for th in spec['coverage_targets']:
        ids=np.flatnonzero(curves[0]>=th)
        hits[str(th)]=dict(sim_s=float(ids[0]*CONTROL_DT),acquisition_wall_s=float(tr['wall_s'][ids[0]])) if len(ids) else None
    result=dict(arm=arm,archived=archive,seconds=spec['seconds'],trace_sha256=src_hash,
                matches_240s_prefix=matches,targets=hits,snapshots=snaps,
                low_two=float((valid[1:].sum(1)>=2).mean()),low_three=float((valid[1:].sum(1)>=3).mean()),
                over40_s=float((load[1:]>40.).sum()*CONTROL_DT))
    save_json(folder/'full-score.json',result)
    print(f'{arm}: {spec["seconds"]} s, coverage {curves[0,-1]:.1%}',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','score'));p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--arm',choices=METHODS[:2]);args=p.parse_args()
    if args.action=='prepare':
        prepare(args.root)
        return
    root=args.root/'pear-full'
    spec=json.loads((root/'long-protocol.json').read_bytes())
    if args.action=='run':
        if args.arm is None:
            p.error('--arm required for a new rollout')
        run(root,'pear','active-02',args.arm,spec['seconds'])
    else:
        torch.set_num_threads(1)
        records={};jobs=list(METHODS)
        while jobs:
            for arm in jobs.copy():
                if arm.startswith('aude') or (root/'runs/pear/active-02'/arm/'run.json').exists():
                    records[arm]=evaluate(args.root,arm,spec)
                    jobs.remove(arm)
            if jobs:
                sleep(4)
        save_json(root/'long-results.json',[records[a] for a in METHODS])


if __name__=='__main__':
    main()
