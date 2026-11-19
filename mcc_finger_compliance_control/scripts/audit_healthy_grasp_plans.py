"""Audit all raw references or probe health-constrained replacements by family."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np

from healthy_grasp_planner import HealthyGraspPlanner,HealthConfig,state_report,DEFAULT_BANK
from plan_waypoint_viewer import PlanReference

CONTEXT=None


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def initialize(bank):
    global CONTEXT
    CONTEXT=HealthyGraspPlanner(bank=Path(bank))


def raw_audit(task):
    started=time.monotonic();c=CONTEXT;r=PlanReference.load(Path(task['plan']))
    rows=[]
    for k,(frame,q) in enumerate(zip(r.frames,r.q)):
        s=c.state(q,r.poses[frame]);g=s['geometry']
        rows.append(dict(frame=int(frame),original_valid=bool(r.valid[k]),
            max_pad_penetration_m=float(np.maximum(-g['distance'],0).max()),
            non_tip_min_m=float(g['non_tip_distance'].min()),
            cross_finger_penetration_m=float(g['self_penetration']),
            health_floor_pass=bool((s['health']>=c.health_floor).all()),
            healthy_joint_bound_pass=bool((s['joint_margin_rad']>=-1e-6).all()),
            health_min_ratio=float((s['health']/c.health_target).min()),
            distal_tip_q_rad=q[[3,7,11,15]].tolist(),
            min_healthy_joint_margin_rad=float(s['joint_margin_rad'].min()),
            nonpenetrating_health_pass=c.valid(s)))
    worst=int(np.argmin([row['health_min_ratio'] for row in rows]))
    diagnostic=[]
    for k in sorted(set([0,worst])):
        s=c.state(r.q[k],r.poses[r.frames[k]],capacity=True)
        diagnostic.append(dict(frame=int(r.frames[k]),**state_report(s)))
    return dict(task,mode='raw_audit',knots=rows,initial=diagnostic[0],capacity_probes=diagnostic,
        initial_penetrating=rows[0]['max_pad_penetration_m']>1e-6,
        accepted_knots_health_fail=sum(row['original_valid'] and not (row['health_floor_pass']
            and row['healthy_joint_bound_pass']) for row in rows),elapsed_s=time.monotonic()-started)


def replacement_probe(task):
    started=time.monotonic();c=CONTEXT;r=PlanReference.load(Path(task['plan']))
    distal=[2,3,6,7,10,11,14,15]
    worst_distal=int(np.argmax(np.maximum(r.q[:,distal]-c.upper[distal],0).max(-1)))
    frames=([0] if task.get('initial_only',False) else
        sorted(set(np.linspace(0,len(r.poses)-1,5).round().astype(int).tolist()+[int(r.frames[worst_distal])])))
    rows=[]
    for frame in frames:
        source=r.sample(frame)[1]
        before=c.state(source,r.poses[frame],capacity=True)
        q,after=c.solve(r.poses[frame],source,multistart=3)
        rows.append(dict(frame=frame,q_source=source.tolist(),q_new=q.tolist(),
            before=state_report(before),after=state_report(after)))
        print(f"[HEALTH-PROBE] {task['candidate_id']:04d} frame={frame} "
              f"valid={after['valid']} gap_mm={np.round(after['geometry']['distance']*1000,3).tolist()}",flush=True)
    return dict(task,mode='replacement_probe',frames=rows,
        passed=sum(row['after']['valid'] for row in rows),tested=len(rows),
        initial_pass=rows[0]['after']['valid'],elapsed_s=time.monotonic()-started,
        scope=('Initial geometric state only; not a physical startup or continuous reference'
            if task.get('initial_only',False) else
            'Independent state probes, not a certified continuous reference or physical rollout'))


def tasks_from_source(root):
    rows=json.loads((root/'collection_manifest.json').read_text())['rows']
    tasks=[]
    for row in sorted(rows,key=lambda r:r['candidate_id']):
        if row.get('attempt_kind')!='physical':continue
        summary=json.loads((Path(row['source_output'])/'summary.json').read_text())
        tasks.append(dict(candidate_id=row['candidate_id'],family=row['family'],
            geometry_group=row['geometry_group'],plan=summary['plan']))
    return tasks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('audit','probe','initial'),required=True)
    p.add_argument('--bank',type=Path,default=DEFAULT_BANK)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--per-family',type=int,default=6)
    p.add_argument('--ids',type=int,nargs='+')
    a=p.parse_args()
    if a.workers<1:p.error('workers must be positive')
    tasks=tasks_from_source(a.source)
    if a.ids is not None:tasks=[t for t in tasks if t['candidate_id'] in a.ids]
    elif a.mode=='probe':
        families=defaultdict(dict)
        for task in tasks:families[task['family']].setdefault(task['geometry_group'],task)
        tasks=[]
        for family,by_group in sorted(families.items()):
            values=list(by_group.values());indices=np.unique(np.linspace(0,len(values)-1,
                min(a.per_family,len(values))).round().astype(int))
            tasks.extend(values[i] for i in indices)
        tasks.sort(key=lambda t:t['candidate_id'])
    if a.mode=='initial':tasks=[dict(t,initial_only=True) for t in tasks]
    a.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(mode=a.mode,tasks=tasks,config=asdict(HealthConfig()),bank=str(a.bank.resolve()),
        bank_sha256=hashlib.sha256(a.bank.read_bytes()).hexdigest(),
        planner_sha256=hashlib.sha256(Path(__file__).with_name('healthy_grasp_planner.py').read_bytes()).hexdigest(),
        physical_compliance='Geometric references never target penetration; physical contact compliance is separate')
    if (a.output/'protocol.json').exists() and json.loads((a.output/'protocol.json').read_text())!=protocol:
        raise RuntimeError('Protocol changed; use fresh output')
    save(a.output/'protocol.json',protocol)
    rows=[json.loads(f.read_text()) for f in (a.output/'results').glob('candidate_*.json')]
    completed={r['candidate_id'] for r in rows}
    started=time.monotonic()
    def publish(phase):
        strata={}
        for family in sorted({t['family'] for t in tasks}):
            group=[r for r in rows if r['family']==family]
            strata[family]=dict(completed=len(group),initial_penetrating=sum(r.get('initial_penetrating',False) for r in group),
                initial_probe_pass=sum(r.get('initial_pass',False) for r in group),
                probe_passed=sum(r.get('passed',0) for r in group),probe_tested=sum(r.get('tested',0) for r in group))
        save(a.output/'summary.json',dict(phase=phase,mode=a.mode,pid=os.getpid(),requested=len(tasks),
            completed=len(rows),elapsed_s=time.monotonic()-started,family_strata=strata,
            rows=sorted(rows,key=lambda r:r['candidate_id'])))
    publish('running')
    action=raw_audit if a.mode=='audit' else replacement_probe
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(str(a.bank),)) as pool:
        futures={pool.submit(action,t):t for t in tasks if t['candidate_id'] not in completed}
        for future in as_completed(futures):
            task=futures[future]
            try:row=future.result()
            except Exception as exc:row=dict(task,error=f'{type(exc).__name__}: {exc}')
            rows.append(row);save(a.output/'results'/f"candidate_{row['candidate_id']:04d}.json",row)
            publish('running')
            print(f"[HEALTH] {a.mode} {len(rows)}/{len(tasks)} candidate={row['candidate_id']}",flush=True)
    publish('completed')


if __name__=='__main__':main()
