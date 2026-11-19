"""Full-length, independently stopped collection force experiments on fixed references."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
DATA = ROOT / 'mcc_finger_compliance_control/data'
ARMS = {'baseline': (1., 'baseline'), 'soft_arm': (.25, 'baseline'),
        'more_mcc': (1., 'outward'), 'both': (.25, 'outward')}


def cases():
    result = [(f'old_{i}', DATA/f'long_palm_oracle_geometry_20261013/plan_{i}.h5',
               f'old_{i}', 'original short-arc-derived long reference')
              for i in (31, 36, 39, 41, 43, 45)]
    result += [(name, DATA/'exploration_20261015/teacher_pilot'/
                f'candidate_{i:04d}/teacher_round0.h5', name, name)
               for name, i in (('meridian', 0), ('spiral', 4), ('direction_sweep', 6))]
    result += [('repaired_orbit', DATA/'exploration_20261015/orbit_clearance_repair/teacher_hard2_preferred5.h5',
                'repaired_orbit', 'clearance-repaired oblique full orbit')]
    result += [(f'grasp_{i}', DATA/f'grasp_clearance_20261013/plan_{i}_full.h5',
                f'old_{i}', 'same palm reference with grasp/clearance optimized posture')
               for i in (39, 45)]
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def analyze_job(job):
    from analyze_grasp_clearance import rollout_report
    directory = Path(job['output'])
    summary = json.loads((directory/'summary.json').read_text())
    z = np.load(directory/'trace.npz')
    start = summary.get('actual_motion_start_step',1200)
    task = z['step']>start if start is not None else np.zeros(len(z['step']),bool)
    if not task.any():
        row = dict(status='initialization_error', error=summary['error'],
                   strict_healthy_four_rate_full=0., found_four_rate_full=0.,
                   observed_frames=0, requested_frames=job['frames'],
                   initial_strict_four_rate=None, complete_and_all_hand_parts_under100N=False,
                   peak_stop_sensor_force_n=0.,strict_per_finger_full=[0.]*4,
                   found_per_finger_full=[0.]*4,longest_observed_four_loss_frames=job['frames'])
    else:
        row, _ = rollout_report(directory, job['frames'])
    initial = ((z['step']>start-50)&(z['step']<=start) if start is not None
               else np.zeros(len(z['step']),bool))
    row['initial_calibrated'] = bool(z['teacher_calibrated'][initial].all()) if initial.any() else False
    row['initial_q'] = z['q'][initial][-1,0].tolist() if initial.any() else None
    row['initial_palm'] = z['palm'][initial][-1,0].tolist() if initial.any() else None
    if summary.get('stable_initialization',False):
        from analyze_collection_initialization import startup_metrics,assess_startup
        row['startup']=assess_startup(startup_metrics(directory))
        row['stable_start']=bool(row['startup']['startup_pass'] and
            row['startup']['posture_stationarity_check'])
    else:
        row['stable_start'] = bool(row['initial_calibrated'] and
            row['initial_strict_four_rate'] is not None and row['initial_strict_four_rate']>=.95)
    audit = json.loads((directory/'independent_force_audit.json').read_text())
    row['force_audit'] = dict(passed=audit['passed'], samples=len(audit['samples']),
        max_vector_error_n=audit['max_vector_error_n'],
        stream_difference_n=audit['max_pre_post_stream_fence_difference_n'],
        failures=len(audit['failures']))
    events = [event for event in audit['events']
              if start is not None and event['control_step']>start]
    if events:
        largest = max(events, key=lambda e:max(e['stop_reference_n']))
        components = np.asarray(largest['components_pad_connector_combined_n'])[0]
        finger, component = np.unravel_index(components.argmax(), components.shape)
        row['independent_peak'] = dict(task_frame=(largest['control_step']-start-1 if start is not None else None),
            substep=largest['physical_substep'], finger=('index','middle','ring','thumb')[finger],
            component=('pad','connector','combined')[component], force_n=float(components.max()),
            object_command_step_mm=largest['object_command_step_mm'][0],
            contacts=largest['contacts'][:4])
    row.update(case=job['case'], arm=job['arm'], geometry_group=job['geometry_group'],
               output=job['output'], elapsed_s=summary['elapsed_s'])
    row['overforce_response']=summary.get('overforce_response','stop')
    from collection_force_events import summarize_force_events
    row['force_events']=summarize_force_events(directory)
    save(directory/'metrics.json', row)
    return row


def aggregate(rows, output):
    groups = {}
    paired = []
    for arm in ARMS:
        selected = [r for r in rows if r['arm']==arm]
        if not selected: continue
        groups[arm] = dict(n=len(selected), stable_starts=sum(r['stable_start'] for r in selected),
            completed=sum(r['status']=='completed' for r in selected),
            force_stops=sum(r['status']=='force_stop' for r in selected),
            strict_four_mean=float(np.mean([r['strict_healthy_four_rate_full'] for r in selected])),
            found_four_mean=float(np.mean([r['found_four_rate_full'] for r in selected])),
            tip_only_four_mean=float(np.mean([r.get('tip_only_four_rate_full',0.) for r in selected])),
            mean_completed_fraction=float(np.mean([r['observed_frames']/r['requested_frames'] for r in selected])),
            median_palm_p95_error_mm=float(np.median([r['palm_position_error_mm']['p95'] for r in selected
                                                    if 'palm_position_error_mm' in r])),
            max_audited_vector_error_n=max(r['force_audit']['max_vector_error_n'] for r in selected),
            audit_passes=sum(r['force_audit']['passed'] for r in selected))
    for name in sorted({r['case'] for r in rows}):
        members = {r['arm']:r for r in rows if r['case']==name}
        if 'baseline' not in members: continue
        b = members['baseline']
        for arm, r in members.items():
            if arm=='baseline': continue
            paired.append(dict(case=name, arm=arm, geometry_group=r['geometry_group'],
                strict_pp=100*(r['strict_healthy_four_rate_full']-b['strict_healthy_four_rate_full']),
                found_pp=100*(r['found_four_rate_full']-b['found_four_rate_full']),
                completed_frames_delta=r['observed_frames']-b['observed_frames'],
                both_stable_start=r['stable_start'] and b['stable_start'],
                initial_q_mae_rad=(float(np.abs(np.asarray(r['initial_q'])-b['initial_q']).mean())
                                   if r['initial_q'] and b['initial_q'] else None),
                initial_palm_translation_difference_mm=(1000*float(np.linalg.norm(
                    np.asarray(r['initial_palm'][:3])-b['initial_palm'][:3]))
                    if r['initial_palm'] and b['initial_palm'] else None)))
    result = dict(scope='Teacher collection, arm-supported palm and mocap object; no DP/RL. 12 references but only 10 independent palm paths; grasp_39/45 share paths with old_39/45.',
        metrics='Existing true-tip strict contact AND found. Strict does not exclude concurrent connector loading or impose a normal-force upper bound. Unexecuted tail counts as failed.',
        pairing='Same source plan and seed. Legacy settling is unchanged; actual initial-state differences are reported. One run per arm/reference; no claim of seed-independent causality.',
        safety='See each row overforce_response: record completes the reference despite force/finite limit excursions; stop uses the old cutoff. 100N remains the event threshold. Numerical failures are not full completion.',
        arms=groups, paired=paired, rows=rows)
    save(output/'analysis.json', result)
    return result


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [c[0] for c in cases() if any(r['case']==c[0] for r in result['rows'])]
    fig, axes = plt.subplots(3,1,figsize=(15,11),sharex=True)
    x=np.arange(len(names)); width=.19
    for k, arm in enumerate(ARMS):
        lookup={r['case']:r for r in result['rows'] if r['arm']==arm}
        for ax, field, scale in zip(axes, ('strict_healthy_four_rate_full','found_four_rate_full','observed_frames'),(100,100,1)):
            values=[lookup[n][field]*scale if n in lookup else np.nan for n in names]
            if field=='observed_frames':
                values=[100*lookup[n]['observed_frames']/lookup[n]['requested_frames'] if n in lookup else np.nan for n in names]
            ax.bar(x+(k-1.5)*width,values,width,label=arm)
    for ax,label in zip(axes,('Strict true-tip four contact (%)','Found four contact (%)','Executed reference (%)')):
        ax.set_ylabel(label); ax.set_ylim(0,105); ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncol=4);axes[-1].set_xticks(x,names,rotation=30,ha='right')
    fig.suptitle('Collection force controls: full-horizon metrics; stopped tail = failure')
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DATA/'force_audit_20261015/full_trajectory')
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--overforce-response',choices=('record','stop'),default='record')
    p.add_argument('--analyze-only',action='store_true')
    args=p.parse_args(); out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    from bundle_manifold_plans import bundle
    sources=cases();jobs=[]
    for i,(name,source,group,family) in enumerate(sources):
        with h5py.File(source) as h:frames=len(h['palm_pose_object'])
        plan=out/'plans'/f'{name}.h5';plan.parent.mkdir(exist_ok=True)
        if not args.analyze_only and not plan.exists():
            with h5py.File(source) as h:
                already_batched = h['grasp_keyframe_q'].ndim==3
            if already_batched:
                shutil.copy2(source,plan)
            else:
                bundle([source],plan)
        if plan.exists():
            with h5py.File(plan) as h:
                if h['grasp_keyframe_q'].ndim!=3 or h['grasp_keyframe_q'].shape[1:]!=(1,16):
                    raise ValueError(f'Invalid single-environment keyframe shape: {plan}')
        for arm in ARMS:
            jobs.append(dict(case=name,source=str(source),source_sha256=sha(source),
                geometry_group=group,family=family,arm=arm,plan=str(plan),frames=frames,
                seed=20261015+i,output=str(out/name/arm)))
    files=['collect_long_oracle_probe.py','independent_contact_force_audit.py',
        'collection_force_experiment.py','collection_force_regulation.py',
        'collect_trajectories.py','surface_mcc_finger.py','true_fingertip_contact.py']
    protocol=dict(jobs=jobs,code_sha256={f:sha(SCRIPTS/f) for f in files},arms=ARMS,
        force_targets_n=[6,6,6,8],arm_pd_baseline=[8000,160],arm_pd_soft=[2000,80],
        outward_cartesian_m=[.006,.010],outward_joint_step_rad=[.025,.080],
        control_dt_s=.01,physics_dt_s=.001,force_event_threshold_n=100,
        overforce_response=args.overforce_response,
        hard_stop_n=100 if args.overforce_response=='stop' else None)
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and not args.analyze_only:
        if json.loads(protocol_path.read_text())!=json.loads(json.dumps(protocol)):
            raise RuntimeError('Protocol/code changed; use a new output directory')
    if not args.analyze_only:save(protocol_path,protocol)
    rows=[]
    def run(job):
        target=Path(job['output']);target.mkdir(parents=True,exist_ok=True)
        metrics=target/'metrics.json'
        if metrics.exists():return json.loads(metrics.read_text())
        if (target/'summary.json').exists():return analyze_job(job)
        scale,authority=ARMS[job['arm']]
        command=[sys.executable,str(SCRIPTS/'collect_long_oracle_probe.py'),
            '--plan',job['plan'],'--output',str(target),'--frames',str(job['frames']),
            '--force-profile','regulated','--force-audit','--force-parts','--substep-force-safety',
            '--arm-stiffness-scale',str(scale),'--mcc-authority',authority,'--seed',str(job['seed']),
            '--overforce-response',args.overforce_response]
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/matplotlib',WARP_CACHE_PATH='/tmp/warp',TQDM_DISABLE='1')
        print(f"[FORCE-SUITE] start {job['case']}/{job['arm']} frames={job['frames']}",flush=True)
        with (target/'run.log').open('w') as log:
            proc=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if not (target/'summary.json').exists():
            raise RuntimeError(f'Infrastructure error rc={proc.returncode}: {target}/run.log')
        row=analyze_job(job)
        if row['status'] in ('execution_error','initialization_error') or (row['observed_frames'] and not row['force_audit']['passed']):
            raise RuntimeError(f'Execution/force audit failed: {target}/metrics.json')
        print(f"[FORCE-SUITE] done {job['case']}/{job['arm']} {row['status']} "
              f"strict={row['strict_healthy_four_rate_full']:.4f} found={row['found_four_rate_full']:.4f}",flush=True)
        return row
    if args.analyze_only:
        rows=[analyze_job(j) for j in jobs if (Path(j['output'])/'summary.json').exists()]
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run,j):j for j in jobs}
            for f in as_completed(futures):
                rows.append(f.result());aggregate(rows,out)
    result=aggregate(rows,out);plot(result,out)
    print(json.dumps(result['arms'],indent=2),flush=True)


if __name__=='__main__':main()
