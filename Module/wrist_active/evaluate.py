"""Audit time-causal wrist decisions and score actual touches against the offline mesh."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
from scipy.sparse.csgraph import dijkstra
import torch
import trimesh

from ..acitive_02.full_signed import signed_gpis as signed_gpis
from ..acitive_02.full_signed import signed_surface_for_output
from ..acitive_02.active_gpis import VOXEL_M
from ..acitive_02.experiment import CONTROL_DT, digest
from ..i04_geometric_traversal import fingertip_r_gpis as ridge
from ..i04_geometric_traversal.contracts import ExperimentConfig
from ..i04_geometric_traversal.fingertip_surface_report import mesh_accuracy, MESH_SAMPLES, METRIC_SEED
from ..i04_geometric_traversal.fixtures import SurfaceGraph
from ..i04_geometric_traversal.oracle import MeasuredCoverage
from .experiment import ACTIVE2, ARMS, BUDGET, CASES, PREVIEW, ROOT, save_json, scenario
from .planner import LOAD_LIMIT


def bouts(mask):
    edges=np.flatnonzero(np.diff(np.r_[False,mask,False]))
    return (edges[1::2]-edges[::2])*CONTROL_DT


def movement(q):
    idx=np.arange(len(q))
    a,b=np.maximum(0,idx-2),np.minimum(len(q)-1,idx+3)
    t=(b-a)*CONTROL_DT
    lin=np.linalg.norm(q[b,:3]-q[a,:3],axis=1)/t
    dot=np.abs(np.einsum('ij,ij->i',q[a,3:7],q[b,3:7]))
    ang=2*np.arccos(np.clip(dot,0.,1.))/t
    return (lin>=.001)|(ang>=.02)


class Reference:
    def __init__(self,surface):
        mesh=trimesh.load_mesh(surface,process=False)
        self.truth=trimesh.sample.sample_surface(mesh,MESH_SAMPLES,seed=METRIC_SEED+1)[0]
        self.graph=SurfaceGraph(mesh,coverage_radius_m=.025)
        self.ledger=MeasuredCoverage(self.graph,ExperimentConfig())

    @lru_cache(maxsize=6000)
    def neighbors(self,vertex):
        dist=dijkstra(self.graph.adjacency,directed=False,indices=vertex,limit=.025)
        return np.flatnonzero(dist<=.010),np.flatnonzero(dist<=.025)

    def area(self,points,normals,ticks,length):
        never=length+1
        registered=np.full(len(self.ledger.vertex_area),never,dtype=np.int32)
        count=0
        for i in range(0,len(ticks),10000):
            sl=slice(i,i+10000)
            v,err=self.graph.oriented_nearest_vertices(points[sl],normals[sl])
            valid=(err<=.005)&(np.einsum('ij,ij->i',self.graph.normals[v],normals[sl])>=.5)
            np.minimum.at(registered,v[valid],ticks[sl][valid])
            count+=valid.sum()
        first=np.full((2,len(registered)),never,dtype=np.int32)
        for v in np.flatnonzero(registered<=length):
            for k,ids in enumerate(self.neighbors(int(v))):
                first[k,ids]=np.minimum(first[k,ids],registered[v])
        weight=self.ledger.vertex_area/self.ledger.vertex_area.sum()
        curves=[np.cumsum(np.bincount(row,weights=weight,minlength=never+1))[:length+1] for row in first]
        return np.minimum(curves,1.),int(count)


@lru_cache(maxsize=3)
def reference(obj):
    return Reference(scenario(obj,'start-01')[1])


def read_trace(path,count):
    with np.load(path,allow_pickle=False) as f:
        keys=('qpos','points_m','normals_outward','pad_valid','pad_force_n','region_load_n','wall_s')
        return {k:f[k][:count+1] for k in (*keys,*[k for k in ('palm_command','published_preview') if k in f])}


def audit(trace,plans,wrist_only):
    length=len(trace['qpos'])-1
    if 'published_preview' in trace:
        np.testing.assert_array_equal(trace['published_preview'][1:length-PREVIEW+2],trace['palm_command'][PREVIEW:])
    valid=trace['pad_valid']
    light=trace['region_load_n'].max(axis=1)<LOAD_LIMIT
    if np.any(valid & (trace['pad_force_n']<.199999)):
        raise ValueError('Trace labels a subthreshold point as a touch')
    support=np.ones_like(valid)
    if not wrist_only:
        return support
    ticks,fingers=np.nonzero(valid&light[:,None])
    xyz=trace['points_m'][ticks,fingers]
    _,indices=np.unique(np.floor(xyz/VOXEL_M).astype(int),axis=0,return_index=True)
    first=np.sort(ticks[indices])
    for i,row in enumerate(plans):
        tick=row['tick']
        n=int(np.searchsorted(first,tick,side='right'))
        last=int(first[n-1]) if n else None
        if row['online_cells']!=n or row['last_new_touch']!=last or row['observed_through']!=tick:
            raise ValueError(f'Noncausal surface receipt at {tick}: {row["online_cells"]}, expected {n}')
        if row['mode']=='scan':
            a=tick+PREVIEW+1
            b=plans[i+1]['tick']+PREVIEW+1 if i+1<len(plans) else len(valid)
            support[a:b]=False
            support[a:b,row['pair']]=True
    return support


def reconstruct(points,normals,ref,output,kind,*,signed_device=None):
    start=perf_counter()
    ids=ridge.voxel_representatives(points) if len(points) else np.array([],dtype=int)
    if len(ids)>ridge.CONTACT_BUDGET:
        ids=ids[ridge.farthest_indices(points[ids],ridge.CONTACT_BUDGET)]
    status='too_few_points'
    device=None
    score=dict(precision=0.,recall=0.,fscore=0.)
    error=None
    if len(ids)>=4:
        p,n=points[ids],normals[ids]
        try:
            if kind=='signed':
                mesh,_,device=signed_surface_for_output(p,n,device=signed_device)
            else:
                bounds=ridge.bounds_for_contacts(p)
                mesh,_=ridge.reconstruct(p,n,bounds,bounds)
        except (ValueError,RuntimeError) as exc:
            if not any(x in str(exc).lower() for x in ('level','surface','ridge','no supported')):
                raise
            status=str(exc)
        else:
            if len(mesh.faces):
                mesh.export(output)
                metrics=mesh_accuracy(mesh,ref.truth)
                score=metrics['at_mm']['5']
                error=metrics['symmetric_mean_mm']
                status='ok'
    result=dict(**score,mean_error_mm=error,status=status,points=len(ids),compute_s=perf_counter()-start)
    if kind=='signed':
        result['gpis_device']=device
    return result


def score(root,obj,start,arm,archive=False):
    torch.set_num_threads(1)
    folder=root/('archive' if archive else 'runs')/obj/start/arm
    folder.mkdir(parents=True,exist_ok=True)
    source=ACTIVE2/arm if archive else folder
    tracefile=source/'trace.npz'
    checksum=digest(tracefile)
    scorepath=folder/'score.json'
    if scorepath.exists():
        old=json.loads(scorepath.read_bytes())
        if old['trace_sha256']==checksum and old['evaluation_sha256']==digest(Path(__file__)):
            return old
    if archive:
        signed=json.loads((ACTIVE2/'signed-results.json').read_bytes())
        assert checksum==next(r['trace_sha256'] for r in signed['runs'] if r['arm']==arm)
        plans=[]
        manifest=json.loads((source/'run.json').read_bytes())
        with np.load(ACTIVE2/'initial.npz') as data:
            from hashlib import sha256
            assert manifest['initial_state_sha256']==sha256(data['integration_state'].tobytes()).hexdigest()
    else:
        receipt=json.loads((folder/'run.json').read_bytes())
        assert receipt['trace_sha256']==checksum and digest(folder/'plans.jsonl')==receipt['plans_sha256']
        plans=[json.loads(line) for line in (folder/'plans.jsonl').read_text(encoding='utf-8').splitlines()]
        assert receipt['executed_s']==BUDGET
    length=round(BUDGET/CONTROL_DT)
    trace=read_trace(tracefile,length)
    assert len(trace['qpos'])==length+1
    ref=reference(obj)
    points,normals=trace['points_m'],trace['normals_outward']
    load=trace['region_load_n'].max(axis=1)
    valid=trace['pad_valid'].astype(bool)
    light=valid&(load<LOAD_LIMIT)[:,None]
    support=audit(trace,plans,not archive and arm!='aude-mcc')
    ticks,fingers=np.nonzero(light)
    xyz,norm=points[ticks,fingers],normals[ticks,fingers]
    curves,registered=ref.area(xyz,norm,ticks,length)
    both_support=(light&support).sum(1)>=2
    extra=(~support[ticks,fingers])&both_support[ticks]
    full_cells=np.unique(np.floor(xyz/VOXEL_M).astype(int),axis=0)
    marginal=None
    if not archive and arm!='aude-mcc':
        retained=np.unique(np.floor(xyz[~extra]/VOXEL_M).astype(int),axis=0)
        marginal=int(len(full_cells)-len(retained))
    moving=movement(trace['qpos'])
    moving[0]=False
    snapshots=[]
    for seconds in (30,60,120,240):
        i=round(seconds/CONTROL_DT)
        visible=ticks<=i
        snap=reconstruct(xyz[visible],norm[visible],ref,folder/f'signed-{seconds}s.ply','signed')
        snapshots.append(dict(seconds=seconds,coverage10=float(curves[0,i]),coverage25=float(curves[1,i]),signed=snap))
    rgpis=reconstruct(xyz,norm,ref,folder/'ridge-final.ply','ridge')
    passed=np.flatnonzero(curves[0]>=.70)
    quality_time=next((s['seconds'] for s in snapshots if s['signed']['fscore']>=.40),None)
    clean2=light.sum(1)>=2
    clean3=light.sum(1)>=3
    bouts3=bouts(clean3[1:])
    plan_ms=[p['planning_ms'] for p in plans]
    result=dict(obj=obj,start=start,arm=arm,archived=archive,seconds=BUDGET,trace_sha256=checksum,
        evaluation_sha256=digest(Path(__file__)),registered=registered,light_touch_rows=len(xyz),
        first_70pct10_sim_s=float(passed[0]*CONTROL_DT) if len(passed) else None,
        first_70pct10_acquisition_wall_s=float(trace['wall_s'][passed[0]]) if len(passed) else None,
        first_observed_f40_sim_s=quality_time,light_two=float(clean2[1:].mean()),light_three=float(clean3[1:].mean()),
        moving_s=float(moving.sum()*CONTROL_DT),moving_light_two=float(clean2[moving].mean()) if moving.any() else None,
        moving_light_three=float(clean3[moving].mean()) if moving.any() else None,
        sustained_light_three=float(bouts3[bouts3>=1.].sum()/BUDGET),
        over40_s=float(np.sum(load[1:]>40.)*CONTROL_DT),peak_n=float(load[1:].max()),
        longest_no_contact_s=float(bouts(~valid[1:].any(1)).max(initial=0.)),
        wrist_travel_m=float(np.linalg.norm(np.diff(trace['qpos'][:,:3],axis=0),axis=1).sum()),
        full_cells=int(len(full_cells)),extra_exclusive_cells=marginal,
        planning_p50_ms=float(np.median(plan_ms)) if plans else None,planning_s=sum(plan_ms)/1000. if plans else None,
        acquisition_wall_s=float(trace['wall_s'][-1]),snapshots=snapshots,ridge=rgpis)
    np.savez_compressed(folder/'coverage-curve.npz',time_s=np.arange(length+1)*CONTROL_DT,coverage10=curves[0],coverage25=curves[1])
    save_json(scorepath,result)
    print(f'{obj}/{start}/{arm} scored, 10mm {curves[0,-1]:.1%}; F5 {snapshots[-1]["signed"]["fscore"]:.1%}',flush=True)
    return result


def score_probes(root):
    from scipy.stats import spearmanr
    groups=[]
    for obj,start in CASES:
        directory=root/'branches'/obj/start
        rows=[]
        for folder in sorted(directory.glob('rank-*')):
            receipt=json.loads((folder/'run.json').read_bytes())
            assert digest(folder/'trace.npz')==receipt['trace_sha256']
            trace=read_trace(folder/'trace.npz',round(receipt['executed_s']/CONTROL_DT))
            plans=[json.loads(x) for x in (folder/'plans.jsonl').read_text(encoding='utf-8').splitlines()]
            audit(trace,plans,True)
            branch=receipt['branch']
            first,last=branch['first_tick'],branch['last_tick']
            clean=trace['pad_valid']&(trace['region_load_n'].max(1)<LOAD_LIMIT)[:,None]
            def cells(mask):
                return set(map(tuple,np.floor(trace['points_m'][mask]/VOXEL_M).astype(int)))
            previous=clean.copy()
            previous[first+1:]=False
            actual=clean.copy()
            actual[:first+1]=actual[last+1:]=False
            pair=np.zeros(4,bool)
            pair[branch['pair']]=True
            anchor=actual&pair
            old=cells(previous)
            both=clean[first+1:last+1][:,pair].all(1)
            rows.append(dict(rank=int(folder.name.split('-')[1]),**branch,
                realized_new_cells=len(cells(actual)-old),realized_extra_cells=len(cells(actual)-old-cells(anchor)),
                support_time_fraction=float(both.mean()),max_load_n=float(trace['region_load_n'][first+1:last+1].max())))
        if len(rows)!=4 or len({(r['state_sha256'],r['prefix_sha256']) for r in rows})!=1 or len({r['choice'] for r in rows})!=4:
            raise ValueError(f'Invalid matched probe group {obj}/{start}')
        gain=np.array([r['predicted_gain'] for r in rows])
        realized=np.array([r['realized_new_cells'] for r in rows])
        rho=float(spearmanr(gain,realized).statistic) if np.ptp(gain)>0 and np.ptp(realized)>0 else None
        groups.append(dict(obj=obj,start=start,rank_correlation=rho,top_rank_tied_or_best=bool(realized[0]==max(realized)),rows=rows))
    save_json(root/'probe-results.json',groups)
    return groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('main','archive','probes'))
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--arm',choices=ARMS)
    args=p.parse_args()
    if args.action=='probes':
        score_probes(args.root)
    elif args.action=='archive':
        for arm in ('aude-mcc','aude-single-all-mcc'):
            score(args.root,'pear','active-02',arm,True)
    else:
        jobs=[(o,s,a) for o,s in CASES for a in ARMS if args.arm is None or args.arm==a]
        jobs.extend(('pear','active-02',a) for a in ('history-dp','history-mcc') if args.arm is None or args.arm==a)
        while jobs:
            for job in jobs.copy():
                if (args.root/'runs'/Path(*job)/'run.json').is_file():
                    score(args.root,*job)
                    jobs.remove(job)
            if jobs:
                sleep(4)


if __name__=='__main__':
    main()
