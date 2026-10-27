"""Continue verified teacher collection with measured region stratification."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np

from prepare_variable_collection_references import balanced_ranges, write_reference
from regional_contact_coverage import SurfaceRegions
from run_leap_hardware_evaluation import analyze, save
from plot_leap_hardware_evaluation import motion_statistics
from run_variable_length_collection import (CONTROL_FILES, execution_contract,
    make_command, raw_palm_slice, rejection_reasons, sha)
from stratified_contact_sampling import (parent_coverage, region_order, region_reached,
    nominal_fingertip_coverage, target_windows, targeted_spec, choose_family, generate_targeted_palm)

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
DATA = SCRIPTS.parent/'data'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bank', type=Path, default=DATA/'dp0_true_tip_1500/long_chains/training_only_healthy_seed_bank.npz')
    p.add_argument('--target', type=int, default=221)
    p.add_argument('--min-frames', type=int, default=800)
    p.add_argument('--max-child-frames', type=int, default=4000)
    p.add_argument('--cell-parent-goal', type=int, default=2)
    p.add_argument('--height-parent-goal', type=int, default=8)
    p.add_argument('--planning-workers', type=int, default=3)
    p.add_argument('--collect-workers', type=int, default=4)
    p.add_argument('--attempts-per-cell', type=int, default=32)
    p.add_argument('--max-new-parents', type=int, default=512)
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--min-free-gb', type=float, default=15.)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    if min(a.target,a.min_frames,a.cell_parent_goal,a.height_parent_goal,
           a.planning_workers,a.collect_workers,a.attempts_per_cell) < 1:
        p.error('Counts and quotas must be positive')
    previous = a.previous.resolve(); out = a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    old_lock=(previous/'coordinator.lock').open()
    try: fcntl.flock(old_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Previous coordinator is still draining; wait before transition')
    lock=(out/'coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ('results','selected','qualified_reserve','rollouts','planning','plans','parents','source_snapshot','shared_models'):
        (out/name).mkdir(exist_ok=True)
    files=tuple(dict.fromkeys((*CONTROL_FILES,Path(__file__).name,'stratified_contact_sampling.py',
        'regional_contact_coverage.py','plan_exploration_dataset.py','generate_manifold_palm_plan.py')))
    hashes={name:sha(SCRIPTS/name) for name in files}
    protocol=dict(config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='prepare_only'},
        source_sha256=hashes,bank_sha256=sha(a.bank),
        controls=json.loads((previous/'protocol.json').read_text())['controls'],
        acceptance=dict(strict_four_min=.95,strict_each_min=.95,max_loss_frames=10,force_max_n=100.,
            stable_initialization=True,complete=True,measured_target_region=True),
        region_definition='Original5 PCA height bands x12 azimuth sectors; >=50 distinct frames per episode',
        coverage_unit='Distinct geometric parents; children, paired speeds and Tube variants do not multiply credit',
        regional_goals=dict(min_parents_per_cell=a.cell_parent_goal,min_parents_per_height=a.height_parent_goal),
        tube=dict(clean_probability=.25,norm_range_rad=[.003,.020],tau_s=.35),
        reference_rule='Try a complete new healthy reference first; after failure collect independent, long child windows; exact original speed',
        off_target_rule='Preserve quality-passing off-target data separately; it does not satisfy the requested regional collection',
        scope='Mustard teacher collection only; no DP/RL or MCC parameter changes')
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:
        raise RuntimeError('Protocol changed: use a new output directory')
    save(out/'protocol.json',protocol)
    for name in files:
        if not (out/'source_snapshot'/name).exists():shutil.copy2(SCRIPTS/name,out/'source_snapshot'/name)
    history=json.loads((previous/'collection_manifest.json').read_text())['rows']
    known={r['name'] for r in history}
    # Workers can finish after SIGINT while the former executor drains. Their
    # already audited physical results must not disappear during transition.
    for path in (previous/'rollouts').glob('*/variable_collection_metrics.json'):
        row=json.loads(path.read_text())
        if row['name'] not in known:history.append(row);known.add(row['name'])
    rows={r['name']:dict(r,imported_previous=True) for r in history}
    for path in (out/'results').glob('*.json'):
        row=json.loads(path.read_text());rows[row['name']]=row
    for row in rows.values():
        if not row.get('selected') or not row.get('imported_previous'):continue
        source=Path(row['selected_file']);dest=out/'selected'/source.name
        if not dest.exists():os.link(source,dest)
        row.update(imported_selected_file=str(source),selected_file=str(dest))
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        MPLCONFIGDIR='/tmp/matplotlib',WARP_CACHE_PATH='/tmp/warp',TQDM_DISABLE='1')
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    from object_catalog import MeshNormalOracle,load_object_config
    from run_exploration_recollection import label_split
    regions=SurfaceRegions(MeshNormalOracle.from_config(load_object_config('ycb_mustard')).vertices)
    started=time.monotonic();active_plan={};active_collect={};queue=[]
    attempts=Counter();attempted_families=Counter();used=set()
    for row in rows.values():
        if row.get('imported_previous'):continue
        if row.get('target_region') is not None:attempts[tuple(row['target_region'])]+=1
        if row.get('spec'):attempted_families[row['family']]+=1
        if row.get('source_reference') and row.get('source_slice'):
            used.add((row['source_reference'],*row['source_slice']))
    pool=[];seen=set()
    physical_refs={str(Path(r['reference']).resolve()) for r in history if r.get('attempt_kind')=='physical' and r.get('reference')}
    references={str(Path(r['reference']).resolve()):r for r in history if r.get('reference')}
    for path in (previous/'planning').glob('*/healthy_reference/reference.h5'):
        if str(path.resolve()) not in references:
            match=re.search(r'candidate_(\d+)',str(path))
            if match:
                cid=int(match.group(1));references[str(path.resolve())]=dict(candidate_id=cid,
                    related_plan_group_id=f'mustard_exploration_seed{a.seed}_geometry{cid//2:04d}',family='unknown')
    for path,row in references.items():
        try:
            nominal=nominal_fingertip_coverage(path,regions)
            with h5py.File(path) as h:
                identity=hashlib.sha256(np.asarray(h['palm_pose_object']).tobytes()+np.asarray(h['q_hand']).tobytes()).hexdigest()
                family=row.get('family','unknown')
                if family=='unknown':family=str(h.attrs.get('exploration_family','meridian_arc'))
            if identity in seen:continue
            seen.add(identity)
            pool.append(dict(path=path,candidate_id=row['candidate_id'],parent=row['related_plan_group_id'],
                family=family,nominal=nominal,whole_tested=path in physical_refs,
                original_full=bool(row.get('kept_whole',False))))
        except (KeyError,ValueError,OSError):continue

    def resources():
        if shutil.disk_usage(out).free<a.min_free_gb*1024**3:raise RuntimeError('Disk below reserved floor')
        for name,digest in hashes.items():
            if sha(SCRIPTS/name)!=digest:raise RuntimeError('Frozen source changed: '+name)

    def good():return [r for r in rows.values() if r.get('selected')]

    def coverage_complete(cov):
        return bool(np.min(cov['cells'])>=a.cell_parent_goal and min(cov['heights'])>=a.height_parent_goal)

    def publish(state='running',error=None):
        qualified=good(); strict=parent_coverage(rows.values());found=parent_coverage(rows.values(),'found')
        def serial(c):return {k:v for k,v in c.items() if k!='cell_parents'}
        save(out/'coverage.json',dict(definition=regions.description(),strict=serial(strict),found=serial(found),
            coverage_goals_met=coverage_complete(strict),
            underrepresented_cells=np.argwhere(np.asarray(strict['cells'])<a.cell_parent_goal).tolist(),
            underrepresented_heights=[b for b,n in enumerate(strict['heights']) if n<a.height_parent_goal],
            target_attempts={f'{b}:{s}':n for (b,s),n in attempts.items()}))
        save(out/'status.json',dict(state=state,phase='region_stratified_collection',coordinator_pid=os.getpid(),
            updated_local=time.strftime('%Y-%m-%d %H:%M:%S'),target=a.target,accepted=len(qualified),
            imported_accepted=sum(r.get('imported_previous',False) for r in qualified),
            new_regional_accepted=sum(not r.get('imported_previous',False) for r in qualified),
            accepted_parent_geometries=len({r['related_plan_group_id'] for r in qualified}),
            accepted_full_long=sum(r.get('kept_whole',False) for r in qualified),
            strict_four_mean=float(np.mean([r['strict_healthy_four_rate_full'] for r in qualified])) if qualified else None,
            found_four_mean=float(np.mean([r['found_four_rate_full'] for r in qualified])) if qualified else None,
            region_parent_counts=strict['cells'],height_parent_counts=strict['heights'],
            coarse_cells_covered=int(np.count_nonzero(strict['cells'])),coverage_goals_met=coverage_complete(strict),
            accepted_by_family=dict(Counter(r['family'] for r in qualified)),
            active_planning=[dict(name=j['name'],target_region=j['target_region']) for j in active_plan.values()],
            active_collection=[dict(name=j['name'],target_region=j['target_region']) for j in active_collect.values()],
            pending_collection=len(queue),reference_pool_size=len(pool),elapsed_s=time.monotonic()-started,
            free_gb=shutil.disk_usage(out).free/1024**3,error=error))
        save(out/'collection_manifest.json',dict(protocol='protocol.json',rows=list(rows.values())))

    def finish(row):
        rows[row['name']]=row;save(out/'results'/f'{row["name"]}.json',row)
        print('[REGIONAL]',row['name'],row.get('target_region'),row['status'],
            f"strict={row.get('strict_healthy_four_rate_full')} found={row.get('found_four_rate_full')}",
            'selected=',row.get('selected',False),'accepted=',len(good()),flush=True)

    def scene_link(folder):
        path=folder/'scene.mjb'
        if not path.exists():return
        shared=out/'shared_models'/f'{sha(path)}.mjb'
        try:os.link(path,shared)
        except FileExistsError:pass
        if path.stat().st_ino!=shared.stat().st_ino:
            temp=folder/'scene.shared.mjb';temp.unlink(missing_ok=True);os.link(shared,temp);os.replace(temp,path)

    def job_for_reference(entry,target,begin,end,name):
        reference=out/'plans'/f'{name}.h5'
        if not reference.exists():write_reference(Path(entry['path']),reference,begin,end,entry['parent'])
        group=int(re.search(r'geometry(\d+)',entry['parent']).group(1))
        tube=bool(np.random.default_rng(a.seed+300001+group).random()>=.25)
        return dict(name=name,candidate_id=entry['candidate_id'],variant='region_stratified',
            reference=str(reference),source_reference=entry['path'],source_slice=[begin,end],
            frames=end-begin,related_plan_group_id=entry['parent'],geometry_group=group,
            split=label_split(group,a.seed),family=entry['family'],tube=tube,
            seed=a.seed+3000000+int(hashlib.sha256(name.encode()).hexdigest()[:7],16),
            output=str(out/'rollouts'/name),target_region=list(target),
            kept_whole=bool(entry.get('original_full') and begin==0 and end==entry['nominal']['frames']))

    def collect(job):
        resources();folder=Path(job['output']);folder.mkdir(parents=True,exist_ok=True)
        digest=sha(job['reference']);command=make_command(job)
        if not (folder/'summary.json').exists():
            save(folder/'launch.json',dict(job,command=command,reference_sha256=digest))
            print('[REGIONAL] collecting',job['name'],'region',job['target_region'],job['frames'],'frames',flush=True)
            with (folder/'run.log').open('w') as log:
                process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
            if not (folder/'summary.json').exists():raise RuntimeError(f'Collector exit{process.returncode}')
        if sha(job['reference'])!=digest:raise RuntimeError('Reference mutated during collection')
        row=analyze(job);row.update(job,attempt_kind='physical',reference_sha256=digest)
        if row['status']=='completed':
            row['offline_motion_statistics']=motion_statistics(folder,row)
            row['peak_all_parts_n']=row['offline_motion_statistics']['peak_all_parts_n']
        row['execution_contract']=execution_contract(folder,row)
        reasons=rejection_reasons(row);row['quality_pass']=not reasons
        data=folder/'collected.h5'
        if data.exists():row['measured_coverage']=regions.measured(data)
        row['target_region_reached']=bool(data.exists() and region_reached(row['measured_coverage'],job['target_region']))
        if not row['target_region_reached']:reasons.append('target_region_not_reached')
        row.update(selected=not reasons,rejection_reasons=reasons)
        if row['selected']:
            dst=out/'selected'/f'{job["name"]}.h5'
            if not dst.exists():shutil.copy2(data,dst)
            with h5py.File(dst,'r+') as h:
                h.attrs.update(related_plan_group_id=job['related_plan_group_id'],
                    feasible_segment_source_plan=job['related_plan_group_id'],
                    exploration_split=job['split'],exploration_family=job['family'],
                    exploration_geometry_group=str(job['geometry_group']),tube_augmented=job['tube'],
                    collection_pass_protocol=str(out/'protocol.json'),collection_target_region=job['target_region'],
                    measured_target_region_reached=True,selection_strict_contact_min=.95,
                    source_collection=str(data),kept_whole_reference=job['kept_whole'])
            row['selected_file']=str(dst)
        elif row['quality_pass']:
            # Keep physically qualified long routes even when the current
            # regional request was missed; they remain reviewable separately.
            reserve=out/'qualified_reserve'/f'{job["name"]}.h5'
            if not reserve.exists():os.link(data,reserve)
            row['qualified_reserve_file']=str(reserve)
        save(folder/'regional_collection_metrics.json',row);scene_link(folder)
        return row

    def plan_parent(task):
        resources();spec=task['spec'];folder=out/'parents'/task['name']
        generated=generate_targeted_palm(spec,folder,env,SCRIPTS)
        if generated['status']!='palm_candidate':
            return dict(task,status='palm_generation_failed',selected=False,rejection_reasons=['palm_generation_failed'])
        source=Path(generated['palm_plan'])
        with h5py.File(source) as h:count=len(h['palm_pose_object'])
        # Full reference first. Only unsuccessful references are shortened.
        ranges=[(0,count)]
        if count>=a.min_frames:
            ranges+=balanced_ranges(0,count,a.max_child_frames,a.min_frames)
            if ranges[-1]==(0,count) and count>=2*a.min_frames:
                ranges+=balanced_ranges(0,count,max(a.min_frames,(count+1)//2),a.min_frames)
            elif a.min_frames<count<2*a.min_frames:
                ranges += [(0,a.min_frames),(count-a.min_frames,count)]
        prepared=[];checked=set()
        for begin,end in ranges:
            if (begin,end) in checked or end-begin<a.min_frames:continue
            checked.add((begin,end))
            child=folder/f'window_{begin}_{end}';child.mkdir(parents=True,exist_ok=True)
            raw=child/'palm.h5';parent=f'mustard_exploration_seed{a.seed}_geometry{spec["geometry_group"]:04d}'
            raw_palm_slice(source,raw,begin,end,parent)
            target=child/'healthy_reference';summary=target/'summary.json'
            if not summary.exists():
                command=[sys.executable,str(SCRIPTS/'replan_healthy_reference.py'),'--input',str(raw),
                    '--output',str(target),'--bank',str(a.bank.resolve())]
                save(child/'launch.json',dict(command=command))
                with (child/'planning.log').open('w') as log:
                    subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            if not summary.exists() or not json.loads(summary.read_text()).get('collection_admissible'):continue
            reference=target/'reference.h5';nominal=nominal_fingertip_coverage(reference,regions)
            if list(task['target_region']) not in nominal['coverage']['visited_cells']:continue
            entry=dict(path=str(reference),nominal=nominal,parent=parent,family=spec['family'],
                candidate_id=spec['candidate_id'],whole_tested=False,original_full=(begin==0 and end==count))
            prepared.append(entry)
            if begin==0 and end==count:break
        return dict(task,status='regional_references_prepared' if prepared else 'targeted_hand_planning_failed',
            selected=False,prepared=prepared,rejection_reasons=[] if prepared else ['no_healthy_reference_in_target_region'])

    def pool_job(target,reserved_parents=()):
        cov=parent_coverage(rows.values());choices=[]
        families=Counter(r['family'] for r in good())
        for index,entry in enumerate(pool):
            b,s=target
            if entry['parent'] in reserved_parents:continue
            if entry['parent'] in cov['cell_parents'][b][s]:continue
            for begin,end in target_windows(entry['nominal'],target,a.min_frames,a.max_child_frames,entry['whole_tested']):
                key=(entry['path'],begin,end)
                if key in used:continue
                choices.append(((families[entry['family']],-(end-begin),index),entry,begin,end));break
        if not choices:return None
        _,entry,begin,end=min(choices,key=lambda x:x[0]);used.add((entry['path'],begin,end))
        identity=hashlib.sha256(f'{entry["path"]}:{begin}:{end}'.encode()).hexdigest()[:12]
        return job_for_reference(entry,target,begin,end,f'pool_b{target[0]}_s{target[1]:02d}_{identity}')

    publish('ready' if a.prepare_only else 'running')
    if a.prepare_only:
        cov=parent_coverage(rows.values());order=region_order(cov,attempts)
        save(out/'preflight.json',dict(imported_accepted=len(good()),reference_pool=len(pool),
            next_target_regions=[list(x) for x in order[:12]],
            next_pool_proposals=[{k:v for k,v in job.items() if k not in ('nominal',)}
                for target in order[:8] if (job:=pool_job(target)) is not None]))
        return
    next_parent=max([r.get('regional_parent_index',-1) for r in rows.values() if not r.get('imported_previous')]+[-1])+1
    issued=0
    try:
        with ThreadPoolExecutor(max_workers=a.planning_workers) as cpu,ThreadPoolExecutor(max_workers=a.collect_workers) as gpu:
            while True:
                resources();cov=parent_coverage(rows.values())
                done=len(good())>=a.target and coverage_complete(cov)
                while queue and len(active_collect)<a.collect_workers:
                    job=queue.pop(0)
                    b,s=job['target_region']
                    if job['related_plan_group_id'] in cov['cell_parents'][b][s]:continue
                    active_collect[gpu.submit(collect,job)]=job
                reservations=[tuple(j['target_region']) for j in [*active_plan.values(),*active_collect.values(),*queue]]
                targets=[x for x in region_order(cov,attempts,reservations,a.cell_parent_goal,a.height_parent_goal)
                    if attempts[x]<a.attempts_per_cell]
                while not done and targets and len(queue)+len(active_collect)<a.collect_workers*2:
                    target=targets.pop(0)
                    reserved_parents={j['related_plan_group_id'] for j in [*active_collect.values(),*queue]}
                    reserved_parents.update(f'mustard_exploration_seed{a.seed}_geometry{j["spec"]["geometry_group"]:04d}' for j in active_plan.values())
                    job=pool_job(target,reserved_parents)
                    if job is not None:
                        attempts[target]+=1;queue.append(job);issued+=1;continue
                    if len(active_plan)>=a.planning_workers or next_parent>=a.max_new_parents:continue
                    family=choose_family(rows.values(),attempted_families)
                    spec=targeted_spec(next_parent,a.seed,target,family)
                    task=dict(name=f'parent_{next_parent:04d}_b{target[0]}_s{target[1]:02d}',
                        regional_parent_index=next_parent,target_region=list(target),spec=spec,family=family)
                    attempts[target]+=1;attempted_families[family]+=1;next_parent+=1;issued+=1
                    active_plan[cpu.submit(plan_parent,task)]=task
                if not active_plan and not active_collect and not queue:
                    state='completed' if done else 'regional_search_budget_exhausted'
                    publish(state);break
                ready,_=wait([*active_plan,*active_collect],timeout=5,return_when=FIRST_COMPLETED)
                for future in ready:
                    planning=future in active_plan;task=(active_plan if planning else active_collect).pop(future)
                    try:
                        result=future.result()
                        if planning:
                            prepared=result.pop('prepared',[])
                            finish(result)
                            for index,entry in enumerate(prepared):
                                pool.append(entry)
                                job=job_for_reference(entry,tuple(task['target_region']),0,entry['nominal']['frames'],task['name']+f'_w{index:02d}')
                                used.add((entry['path'],0,entry['nominal']['frames']));queue.append(job)
                        else:
                            finish(result)
                            # A complete long, geometrically valid reference
                            # that fails physics can still yield new starts.
                            if not result.get('selected'):
                                for entry in pool:
                                    if (entry['path']==task['source_reference'] and
                                            task['source_slice']==[0,entry['nominal']['frames']]):entry['whole_tested']=True
                    except Exception as error:
                        finish(dict(task,status='planning_error' if planning else 'collection_error',selected=False,
                            error=f'{type(error).__name__}: {error}',rejection_reasons=['planner_error' if planning else 'collector_error']))
                publish()
    except BaseException as error:
        publish('interrupted',f'{type(error).__name__}: {error}');raise


if __name__=='__main__':main()
