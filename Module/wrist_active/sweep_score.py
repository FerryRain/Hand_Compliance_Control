"""Score sweeps and archived baselines on identical physical time prefixes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import Reference, bouts, movement, reconstruct
from .experiment import ACTIVE2, ROOT, digest, save_json, scenario
from .sweep import DT, LOAD_LIMIT
from .sweep_experiment import OUT, SMOOTH, GENERATED
from ..acitive_02.full_signed import signed_output_device


def ref_mesh(obj,start):
    if obj in SMOOTH:
        return GENERATED/'i01_a_smooth_20261121/runs'/obj/start/'shared/fixed-dp_r01/surface.obj'
    return scenario(obj,start)[1]


def distribution(values, deadline):
    return dict(n=len(values),p50=float(np.median(values)),p95=float(np.percentile(values,95)),
                p99=float(np.percentile(values,99)),maximum=float(np.max(values)),
                deadline_ms=deadline,over=int(np.count_nonzero(values>deadline)))


def recoveries(trace, cells, low_two):
    phases = trace['phase']
    starts = np.flatnonzero((phases[1:] == 1) & (phases[:-1] != 1)) + 1
    events = []
    for j,s in enumerate(starts):
        # Require 200 ms continuously with two light pads after unloading.
        next_start = starts[j+1] if j+1 < len(starts) else len(phases)
        stop = min(s+601,next_start,len(phases))
        scan = np.flatnonzero(phases[s:stop] == 0)
        end = int(s+scan[0]) if len(scan) else None
        t = np.arange(s,stop)
        can = (phases[t]==0) | (phases[t]==3)
        streak = np.convolve((can & low_two[t]).astype(int),np.ones(20,int),'valid') if len(t)>=20 else []
        ok = np.flatnonzero(np.asarray(streak) == 20)
        restored = int(s+ok[0]+19) if len(ok) else None
        cutoff = min(stop,(end if end is not None else s)+500)
        new = int(np.searchsorted(cells,cutoff)-np.searchsorted(cells,s))
        events.append(dict(start_s=float(s*DT),end_s=None if end is None else end*DT,
                           restored_two_s=None if restored is None else restored*DT,new_cells=new))
    return events


def score(source, dest, obj, start, times, *, fast=False):
    dest.mkdir(parents=True,exist_ok=True)
    trace_path = source/'trace.npz'
    checksum = digest(trace_path)
    output = dest/('coverage-only.json' if fast else 'scores.json')
    spec = dict(trace_sha256=checksum,scorer_sha256=digest(Path(__file__)),
                shared_evaluator_sha256=digest(Path(__file__).with_name('evaluate.py')),
                times=times,mesh_sha256=digest(ref_mesh(obj,start)))
    if not fast:
        model_dir=Path(__file__).resolve().parents[1]/'acitive_02'
        spec.update(signed_device=signed_output_device(),signed_model_sha256=digest(model_dir/'full_signed.py'),
                    signed_cuda_sha256=digest(model_dir/'signed_cuda.py'))
    if output.exists() and json.loads(output.read_bytes())['spec']==spec:
        return json.loads(output.read_bytes())
    manifest = json.loads((source/'run.json').read_bytes())
    if manifest.get('trace_sha256') and manifest['trace_sha256'] != checksum:
        raise ValueError('Trace checksum mismatch')
    if source.parent == ACTIVE2:
        archive = json.loads((ACTIVE2/'signed-results.json').read_bytes())
        if checksum != next(r['trace_sha256'] for r in archive['runs'] if r['arm']==source.name):
            raise ValueError('Archived trace checksum mismatch')
    end = round(max(times)/DT)
    with np.load(trace_path,allow_pickle=False) as f:
        keys = ('qpos','region_load_n','pad_valid','points_m','normals_outward','pad_force_n')
        tr = {k:f[k][:end+1] for k in (*keys,*[k for k in ('phase','palm_command','published_preview') if k in f])}
    if len(tr['qpos']) != end+1:
        raise ValueError('Cannot score beyond the physical trace')
    peak = tr['region_load_n'].max(1)
    valid = tr['pad_valid'] & (peak<LOAD_LIMIT)[:,None]
    if np.any(valid & (tr['pad_force_n']<.199999)):
        raise ValueError('Invalid fingertip force')
    ticks,fingers = np.nonzero(valid)
    xyz = tr['points_m'][ticks,fingers]
    normal = tr['normals_outward'][ticks,fingers]
    _,first = np.unique(np.floor(xyz/.0045).astype(int),axis=0,return_index=True)
    cell_ticks = np.sort(ticks[first])
    ref = Reference(ref_mesh(obj,start))
    curves,registered = ref.area(xyz,normal,ticks,end)
    if np.any(np.diff(curves,axis=1)<-1e-12):
        raise ValueError('Coverage decreased')
    moving = movement(tr['qpos'])
    snapshots = []
    for s in times:
        k = round(s/DT)
        count = valid[1:k+1].sum(1)
        sustained = bouts(count>=3)
        touch_end = np.searchsorted(ticks,k,side='right')
        segment = moving[1:k+1]
        snap = dict(s=s,coverage=float(curves[0,k]),coverage25=float(curves[1,k]),
                    mean_coverage=float(curves[0,1:k+1].mean()),cells=int(np.searchsorted(cell_ticks,k,side='right')),
                    low2=float((count>=2).mean()),low3=float((count>=3).mean()),
                    low3_s=float((count>=3).sum()*DT),sustained3=float(sustained[sustained>=1.].sum()/s),
                    moving_low3=float((count[segment]>=3).mean()) if segment.any() else 0.,
                    over8_s=float(np.sum(peak[1:k+1]>=8.)*DT), over40_s=float(np.sum(peak[1:k+1]>40.)*DT),
                    peak_n=float(peak[1:k+1].max()))
        if not fast:
            snap['signed'] = reconstruct(xyz[:touch_end],normal[:touch_end],ref,dest/f'signed-{s:g}s.ply','signed')
        snapshots.append(snap)
    targets = {}
    for th in (50,70,90,95,99):
        ix = np.flatnonzero(curves[0]>=th/100.)
        targets[str(th)] = float(ix[0]*DT) if len(ix) else None
    result = dict(obj=obj,start=start,arm=source.name,spec=spec,registered=registered,
                  thresholds_s=targets,snapshots=snapshots)
    if 'phase' in tr:
        plans = [json.loads(l) for l in (source/'plans.jsonl').read_text(encoding='utf-8').splitlines()]
        if digest(source/'plans.jsonl') != manifest['plan_sha256']:
            raise ValueError('Planner log checksum mismatch')
        for r in plans:
            if r['tick']>end:
                break
            size = np.searchsorted(cell_ticks,r['tick'],side='right')
            if size != r['cells'] or (size and cell_ticks[size-1]!=r['last_new_tick']):
                raise ValueError(f'Noncausal map at {r["tick"]}')
        events = recoveries(tr,cell_ticks,valid.sum(1)>=2)
        save_json(dest/'recoveries.json',events)
        result['recovery'] = dict(attempts=len(events),two=sum(e['restored_two_s'] is not None for e in events),
                                  new_cells=sum(e['new_cells']>0 for e in events))
        # The 200 ms policy preview forecasts the raw wrist and force servo. Future
        # force measurements and fresh plans may alter the realized reference.
        forecast = tr['published_preview'][1:-20]
        realized = tr['palm_command'][21:]
        err = np.linalg.norm(forecast[:,:3]-realized[:,:3],axis=1)*1000
        turn = np.rad2deg(2*np.arccos(np.clip(abs(np.sum(forecast[:,3:]*realized[:,3:],axis=1)),0,1)))
        result['preview_error_p50_p95'] = dict(mm=np.percentile(err,[50,95]).tolist(),
                                               degrees=np.percentile(turn,[50,95]).tolist())
        with np.load(source/'control-timing.npz') as d:
            timing = dict(plan=distribution(d['plan_ms'],200),loop_service=distribution(d['service_ms'],10))
            if len(d['finish_from_release_ms']):
                timing['finish_from_release']=distribution(d['finish_from_release_ms'],10)
        if (source/'timing.npz').exists():
            with np.load(source/'timing.npz') as d:
                timing['dp_rpc']=distribution(d['chunk_rpc_ms'],50)
        result['timing']=timing
    np.savez_compressed(dest/'coverage.npz',time_s=np.arange(end+1)*DT,c10=curves[0],c25=curves[1])
    save_json(output,result)
    print(f'{obj}/{source.name} {max(times)}s | 10mm {snapshots[-1]["coverage"]:.2%} | '
          f'light3 {snapshots[-1]["low3"]:.2%} | t95 {targets["95"]}',flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=OUT)
    p.add_argument('--series',default='runs')
    p.add_argument('--object',default='pear')
    p.add_argument('--start',default='active-02')
    p.add_argument('--arm',default='sweep-dp')
    p.add_argument('--times',nargs='+',type=float,default=[60,120,240])
    p.add_argument('--baseline',choices=('active2','old'))
    p.add_argument('--fast',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1)
    suffix=Path(args.object)/args.start/args.arm
    dest=args.root/args.series/suffix
    if args.baseline == 'active2':
        source=ACTIVE2/args.arm
    elif args.baseline == 'old':
        base=ROOT
        if args.object=='pear' and max(args.times)>240:
            base=ROOT/'pear-full'
        elif args.object in SMOOTH:
            base=ROOT/'smooth-transfer'
        source=base/'runs'/suffix
    else:
        source=dest
    score(source,dest,args.object,args.start,sorted(set(args.times)),fast=args.fast)


if __name__=='__main__':
    main()
