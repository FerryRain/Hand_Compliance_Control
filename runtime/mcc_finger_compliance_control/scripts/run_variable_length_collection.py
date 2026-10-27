"""Collect full qualified references and independently initialized children.

Uses the accepted 10N/handover-v2 teacher chain. A failed long reference can
produce new child references, but only complete independent child rollouts
enter selected/. Every original candidate is attempted before replenishment.
"""
from __future__ import annotations

import argparse
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
from run_leap_hardware_evaluation import analyze, save
from plot_leap_hardware_evaluation import motion_statistics

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
DATA = SCRIPTS.parent/'data'
CONTROL_FILES = (
    'run_variable_length_collection.py', 'prepare_variable_collection_references.py',
    'collect_leap_force_matched.py', 'collect_leap_hardware_probe.py',
    'collect_long_oracle_probe.py', 'collect_trajectories.py',
    'surface_mcc_finger.py', 'collection_initialization.py',
    'collection_force_regulation.py', 'held_relative_target.py',
    'healthy_grasp_planner.py', 'replan_healthy_reference.py',
    'collect_regional_healthy_reference.py', 'collect_healthy_reference.py',
    'true_fingertip_contact.py', 'leap_hardware_profile.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_contract(folder, row):
    """Check every motion physics tick, including the first tick."""
    if row.get('status') != 'completed':
        return dict(passed=False, reason='No complete motion')
    summary = json.loads((folder/'summary.json').read_text())
    start = summary['actual_motion_start_step']
    with np.load(folder/'trace.npz') as z:
        active = (z['step'] > start) & (z['step'] <= start+row['requested_frames'])
        forces = bool(np.allclose(z['force_setpoint'][active, 0], 10.))
    with np.load(folder/'physics_trace.npz') as z:
        active = (z['step'] >= start) & (z['step'] < start+row['requested_frames'])
        sub = z['substep'][active]
        ticks = z['step'][active]
        ctrl = z['ctrl'][active, 0, -16:]
        torque = z['actuator_force'][active, 0, -16:]
    expected = int(row['requested_frames'])
    if len(sub) != expected*10 or not np.array_equal(sub.reshape(-1, 10),
            np.broadcast_to(np.arange(1, 11), (expected, 10))):
        return dict(passed=False, reason='Incomplete physical substeps')
    held = float(np.ptp(ctrl.reshape(expected, 10, 16), axis=1).max())
    with h5py.File(folder/'collected.h5') as h:
        aligned = np.array_equal(ticks[::10], h['episode_step'][:, 0])
        command_error = float(abs(ctrl[::10]-h['q_cmd'][:, 0]).max())
        finite = all(np.isfinite(h[name][:]).all() for name in
                     ('q_hand', 'q_cmd', 'qvel', 'palm_command_pose_object'))
    peak = float(abs(torque).max())
    passed = bool(forces and aligned and finite and held <= 1e-6
        and command_error <= 1e-6 and peak <= .8001
        and row.get('force_audit', {}).get('passed'))
    return dict(passed=passed, frames=expected, force_targets_verified=forces,
        logged_command_max_error_rad=command_error, held_target_max_error_rad=held,
        max_motor_torque_nm=peak, finite_training_state=bool(finite))


def rejection_reasons(row, contact=.95, max_loss=10):
    reasons = []
    if row.get('status') != 'completed': reasons.append('incomplete_execution')
    if not row.get('stable_start'): reasons.append('unstable_initialization')
    if row.get('strict_healthy_four_rate_full', 0.) < contact: reasons.append('strict_four_below95')
    if min(row.get('strict_per_finger_full', [0.]*4)) < contact: reasons.append('strict_finger_below95')
    if row.get('longest_observed_four_loss_frames', max_loss+1) > max_loss:
        reasons.append('persistent_contact_loss')
    if not row.get('execution_contract', {}).get('passed'): reasons.append('execution_contract_failed')
    physics = row.get('offline_motion_statistics')
    if physics is None: reasons.append('missing_motion_physics')
    elif physics['over100_ms'] > 0: reasons.append('over100N_during_motion')
    if not np.isfinite(row.get('peak_all_parts_n', 0.)): reasons.append('nonfinite_force')
    return reasons


def make_command(job):
    command = [sys.executable, '-u', str(SCRIPTS/'collect_leap_force_matched.py'),
        '--leap-profile', 'peak', '--leap-kp', '35', '--leap-kd', '.2',
        '--mcc-force-targets', '10', '10', '10', '10', '--recovery-handover',
        '--object-substep-interpolation', '--plan', job['reference'], '--output', job['output'],
        '--frames', str(job['frames']), '--stable-initialization', '--initialization-extra-steps', '1200',
        '--force-profile', 'regulated', '--force-audit', '--force-parts',
        '--substep-force-safety', '--overforce-response', 'record', '--seed', str(job['seed'])]
    if job['tube']:
        command.extend(['--execution-randomization', '--execution-clean-env-fraction', '0'])
    return command


def raw_palm_slice(source, output, begin, end, group):
    """Keep exact palm geometry/timing; old teacher joints are seeds only."""
    if output.exists(): return
    with h5py.File(source) as h:
        count = len(h['palm_pose_object'])
        if not 0 <= begin < end <= count: raise ValueError('Invalid raw palm interval')
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, 'w') as dst:
            for key, value in h.attrs.items():
                if not key.startswith(('healthy_reference', 'finger_contact', 'grasp_keyframe')):
                    dst.attrs[key] = value
            for key, ds in h.items():
                if key.startswith(('grasp_keyframe', 'fingertip_', 'finger_')) or key == 'non_tip_geom_name':
                    continue
                if ds.ndim and ds.shape[0] == count:
                    if key == 'q_hand': value = np.repeat(ds[begin:begin+1], end-begin, axis=0)
                    elif key == 'qvel': value = np.zeros((end-begin, *ds.shape[1:]), ds.dtype)
                    elif key in ('record_step', 'episode_id'):
                        value = np.arange(end-begin).reshape(-1, *([1]*(ds.ndim-1))) if key == 'record_step' else np.zeros((end-begin, *ds.shape[1:]), ds.dtype)
                    else: value = ds[begin:end]
                    dst.create_dataset(key, data=value, dtype=ds.dtype)
                else: h.copy(key, dst)
            dst.attrs.update(healthy_reference_admissible=False, collection_admissible=False,
                finger_contact_plan=False, subtrajectory_source=str(source.resolve()),
                subtrajectory_source_slice=np.asarray([begin, end], dtype=np.int64),
                related_plan_group_id=group, subtrajectory_timing='Exact source samples and control_dt',
                subtrajectory_supervision='Initial joint seed only; independently replan then physically collect')
            np.testing.assert_array_equal(dst['palm_pose_object'][:], h['palm_pose_object'][begin:end])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed-manifest', type=Path, required=True)
    p.add_argument('--source', type=Path, default=DATA/'exploration_collection_pass_20261016')
    p.add_argument('--reference-caches', type=Path, nargs='+', default=[
        DATA/'healthy_exploration_validation_20261016', DATA/'healthy_regional_collection_20261016'])
    p.add_argument('--bank', type=Path, default=DATA/'dp0_true_tip_1500/long_chains/training_only_healthy_seed_bank.npz')
    p.add_argument('--target', type=int, default=221)
    p.add_argument('--min-frames', type=int, default=800)
    p.add_argument('--max-child-frames', type=int, default=2000)
    p.add_argument('--planning-workers', type=int, default=3)
    p.add_argument('--collect-workers', type=int, default=4)
    p.add_argument('--max-new-parents', type=int, default=512)
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--min-free-gb', type=float, default=15.)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--seed-only', action='store_true')
    a = p.parse_args()
    if min(a.target, a.min_frames, a.planning_workers, a.collect_workers) < 1 or a.max_child_frames < a.min_frames:
        p.error('Invalid counts or frame limits')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (out/'coordinator.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ('results', 'selected', 'rollouts', 'planning', 'plans', 'shared_models', 'new_parents'):
        (out/name).mkdir(exist_ok=True)
    seed_manifest = json.loads(a.seed_manifest.read_text())
    hashes = {n: sha(SCRIPTS/n) for n in CONTROL_FILES}
    protocol = dict(config={k:str(v.resolve()) if isinstance(v, Path) else
        [str(x.resolve()) for x in v] if k == 'reference_caches' else v
        for k,v in vars(a).items() if k != 'prepare_only'}, source_sha256=hashes,
        bank_sha256=sha(a.bank), seed_jobs_sha256=hashlib.sha256(json.dumps(
            [{k:j[k] for k in ('name','reference','frames','command')} for j in seed_manifest['jobs']],
            sort_keys=True).encode()).hexdigest(),
        controls=dict(force_targets_n=[10.]*4, motor_cap_nm=.8, kp=35., kd=.2,
            handover_revision=2, control_hz=100, physics_hz=1000, overforce_response='record'),
        acceptance=dict(strict_four_min=.95, strict_each_min=.95, max_loss_frames=10,
            all_hand_force_max_n=100., independent_initialization=True, full_child_execution=True),
        short_segments='Drop below minimum frames; keep qualifying full long references',
        split_rule='All children, Tube variants, and paired speeds of one geometric parent share one split',
        scene_storage='Lossless hard-link deduplication by SHA256; raw logs and physical traces retained',
        scope='Teacher collection only; no DP/RL parameter updates')
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text()) != protocol:
        raise RuntimeError('Frozen collection protocol changed; use a fresh output directory')
    save(out/'protocol.json', protocol)
    for n in (*CONTROL_FILES, Path(__file__).name, 'prepare_variable_collection_references.py'):
        dst = out/'source_snapshot'/n; dst.parent.mkdir(exist_ok=True)
        if not dst.exists(): shutil.copy2(SCRIPTS/n, dst)
    rows = {q.stem:json.loads(q.read_text()) for q in (out/'results').glob('*.json')}
    env = os.environ.copy(); env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', MPLCONFIGDIR='/tmp/matplotlib', WARP_CACHE_PATH='/tmp/warp', TQDM_DISABLE='1')
    env['PYTHONPATH'] = str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH', '')
    from regional_contact_coverage import SurfaceRegions
    from object_catalog import MeshNormalOracle, load_object_config
    from plan_exploration_dataset import candidate, attempt
    from run_exploration_recollection import label_split
    regions = SurfaceRegions(MeshNormalOracle.from_config(load_object_config('ycb_mustard')).vertices)
    originals = {r['candidate_id']:r for r in json.loads((a.source/'collection_manifest.json').read_text())['rows']}
    settings = argparse.Namespace(seed=a.seed, speeds_m_s=[.03,.05])
    started = time.monotonic(); active_plan = {}; active_collect = {}; phase = 'prepared'

    def accepted(): return [r for r in rows.values() if r.get('selected')]

    def resources():
        if shutil.disk_usage(out).free < a.min_free_gb*1024**3: raise RuntimeError('Disk below reserved floor')
        for name, digest in hashes.items():
            if sha(SCRIPTS/name) != digest: raise RuntimeError('Frozen controller/planner changed: '+name)

    def publish(state='running', error=None):
        good = accepted(); height = np.zeros(5, int); cells = np.zeros((5,12), int); families = {}
        failures = {}
        for r in good:
            families[r['family']] = families.get(r['family'],0)+1
            cov = r['measured_coverage']['strict']
            for b in cov['visited_height_bands']: height[b] += 1
            for b,s in cov['visited_cells']: cells[b,s] += 1
        for r in rows.values():
            for reason in r.get('rejection_reasons',[]): failures[reason] = failures.get(reason,0)+1
        save(out/'status.json', dict(state=state, phase=phase, coordinator_pid=os.getpid(),
            updated_local=time.strftime('%Y-%m-%d %H:%M:%S'), target=a.target,
            accepted=len(good), accepted_parent_geometries=len({r['related_plan_group_id'] for r in good}),
            attempted=len(rows), physical_attempts=sum(r.get('attempt_kind')=='physical' for r in rows.values()),
            accepted_full_long=sum(r.get('kept_whole',False) for r in good),
            strict_four_mean=float(np.mean([r['strict_healthy_four_rate_full'] for r in good])) if good else None,
            found_four_mean=float(np.mean([r['found_four_rate_full'] for r in good])) if good else None,
            accepted_by_family=families, rejection_reasons=failures,
            active_planning=[j['name'] for j in active_plan.values()],
            active_collection=[j['name'] for j in active_collect.values()],
            elapsed_s=time.monotonic()-started, free_gb=shutil.disk_usage(out).free/1024**3, error=error))
        save(out/'coverage.json', dict(definition=regions.description(), height_counts=height.tolist(),
            height_azimuth_counts=cells.tolist(), uncovered_cells=np.argwhere(cells==0).tolist()))
        save(out/'collection_manifest.json', dict(protocol='protocol.json', rows=list(rows.values())))

    def base(cid, name, **values):
        spec = candidate(cid, settings)
        group = f'mustard_exploration_seed{a.seed}_geometry{spec["geometry_group"]:04d}'
        tube = bool(np.random.default_rng(a.seed+300001+spec['geometry_group']).random() >= .25)
        return dict(candidate_id=cid, name=name, case=name, variant='independent_subtrajectory',
            family=spec['family'], geometry_group=spec['geometry_group'], related_plan_group_id=group,
            split=label_split(spec['geometry_group'], a.seed), tube=tube,
            seed=a.seed+2000000+cid*1000+int(hashlib.sha256(name.encode()).hexdigest()[:6],16),
            output=str(out/'rollouts'/name), **values)

    def finish(row):
        rows[row['name']] = row; save(out/'results'/f'{row["name"]}.json', row)
        print(f'[VARIABLE] {row["name"]} {row["status"]} '
            f'strict={row.get("strict_healthy_four_rate_full")} found={row.get("found_four_rate_full")} '
            f'selected={row.get("selected",False)} accepted={len(accepted())}/{a.target}', flush=True)

    def deduplicate_scene(folder):
        path = folder/'scene.mjb'
        if not path.exists(): return
        digest = sha(path); shared = out/'shared_models'/f'{digest}.mjb'
        shared.parent.mkdir(exist_ok=True)
        try: os.link(path, shared)
        except FileExistsError: pass
        if path.stat().st_ino != shared.stat().st_ino:
            if sha(shared) != digest: raise RuntimeError('Model content-address collision')
            tmp = folder/'scene.shared.mjb'; tmp.unlink(missing_ok=True)
            os.link(shared, tmp); os.replace(tmp, path)

    def collect(job):
        resources(); job = dict(job); folder = Path(job['output']); folder.mkdir(parents=True, exist_ok=True)
        command = job.get('command') or make_command(job)
        digest = sha(job['reference'])
        if not (folder/'summary.json').exists():
            save(folder/'launch.json', dict(job, command=command, reference_sha256=digest))
            print('[VARIABLE] collecting',job['name'],job['frames'],'frames',flush=True)
            with (folder/'run.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, timeout=2400)
            if not (folder/'summary.json').exists(): raise RuntimeError(f'Collector exit {result.returncode}: {folder}/run.log')
        if sha(job['reference']) != digest: raise RuntimeError('Reference changed during collection')
        row = analyze(job)
        row.update(job, attempt_kind='physical', reference_sha256=digest)
        if row.get('status') == 'completed':
            row['offline_motion_statistics'] = motion_statistics(folder, row)
            row['peak_all_parts_n'] = row['offline_motion_statistics']['peak_all_parts_n']
        row['execution_contract'] = execution_contract(folder, row)
        row['rejection_reasons'] = rejection_reasons(row)
        row['selected'] = not row['rejection_reasons']
        data = folder/'collected.h5'
        if data.exists(): row['measured_coverage'] = regions.measured(data)
        if row['selected']:
            dst = out/'selected'/f'{job["name"]}.h5'; dst.parent.mkdir(exist_ok=True)
            if not dst.exists(): shutil.copy2(data, dst)
            with h5py.File(dst, 'r+') as h:
                h.attrs.update(related_plan_group_id=job['related_plan_group_id'],
                    feasible_segment_source_plan=job['related_plan_group_id'],
                    exploration_geometry_group=str(job['geometry_group']), exploration_split=job['split'],
                    exploration_family=job['family'], tube_augmented=job['tube'],
                    collection_pass_protocol=str(out/'protocol.json'), healthy_reference=job['reference'],
                    selection_strict_contact_min=.95, source_collection=str(data),
                    kept_whole_reference=job.get('kept_whole',False),
                    measured_region_coverage_json=json.dumps(row['measured_coverage']))
            row['selected_file'] = str(dst)
        save(folder/'variable_collection_metrics.json', row)
        deduplicate_scene(folder)
        return row

    def plan(job):
        resources(); job = dict(job)
        if job.get('reference'): return job
        folder = out/'planning'/job['name']; folder.mkdir(parents=True, exist_ok=True)
        raw = folder/'palm.h5'
        raw_palm_slice(Path(job['source']), raw, *job['source_slice'], job['related_plan_group_id'])
        target = folder/'healthy_reference'; summary = target/'summary.json'
        if not summary.exists():
            command = [sys.executable, str(SCRIPTS/'replan_healthy_reference.py'),
                '--input',str(raw),'--output',str(target),'--bank',str(a.bank.resolve())]
            save(folder/'launch.json',dict(command=command,source_sha256=sha(job['source'])))
            with (folder/'planning.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, timeout=1800)
            if not summary.exists(): raise RuntimeError(f'Planner exit {result.returncode}: {folder}/planning.log')
        info = json.loads(summary.read_text())
        if not info.get('collection_admissible'):
            return dict(job,status='planning_rejected',attempt_kind='planning',selected=False,
                rejection_reasons=['healthy_continuation_search_failed'],planning_summary=str(summary))
        job['reference'] = str(target/'reference.h5')
        return job

    def execute(jobs):
        nonlocal active_plan, active_collect
        todo = [j for j in jobs if j['name'] not in rows]; queue = []; index = 0
        with ThreadPoolExecutor(max_workers=a.planning_workers) as cpu, ThreadPoolExecutor(max_workers=a.collect_workers) as gpu:
            while index < len(todo) or queue or active_plan or active_collect:
                resources()
                while index < len(todo) and len(active_plan) < a.planning_workers and len(queue) < a.collect_workers*2:
                    job = todo[index]; index += 1; active_plan[cpu.submit(plan,job)] = job
                while queue and len(active_collect) < a.collect_workers:
                    job = queue.pop(0); active_collect[gpu.submit(collect,job)] = job
                ready,_ = wait([*active_plan,*active_collect], timeout=5, return_when=FIRST_COMPLETED)
                for future in ready:
                    planning = future in active_plan
                    job = (active_plan if planning else active_collect).pop(future)
                    try:
                        row = future.result()
                        if planning and row.get('reference'): queue.append(row)
                        else: finish(row)
                    except Exception as error:
                        finish(dict(job,status='planning_error' if planning else 'collection_error',
                            selected=False, error=f'{type(error).__name__}: {error}',
                            rejection_reasons=['planner_error' if planning else 'collector_error']))
                publish()

    # Existing full-qualified references are retained using their already
    # collected full data; copying the same path is not a new experiment.
    pilot_root = a.seed_manifest.parent
    original_report = json.loads((pilot_root.parent/'final_comparison/final_comparison.json').read_text())
    full_rows = {r['candidate_id']:r for r in original_report['rows'] if r['variant']=='10N handover v2'}
    seed_jobs = []
    for item in seed_manifest['jobs']:
        cid = item['candidate_id']; job = base(cid,item['name'],reference=item['reference'],
            frames=item['frames'],source_slice=item['source_slice'],kept_whole=item['preserved_full_qualified_reference'])
        job.update(tube=item['tube'],seed=item['seed'],output=item['output'],command=item['command'])
        if job['frames'] < a.min_frames:
            if job['name'] not in rows:
                finish(dict(job,status='short_segment_excluded',selected=False,rejection_reasons=['below_minimum_frames']))
            continue
        if job['kept_whole']:
            old = full_rows[cid]
            job.update(output=old['output'], reference=json.loads((Path(old['output'])/'summary.json').read_text())['plan'],
                seed=old['seed'],tube=old['tube'],reused_full_qualified=True)
            job.pop('command',None)
        seed_jobs.append(job)
    save(out/'prepared_seed_jobs.json', dict(jobs=seed_jobs, minimum_frames=a.min_frames))
    publish('ready' if a.prepare_only else 'running')
    if a.prepare_only: return

    try:
        phase = 'prepared_references_first_pass'; execute(seed_jobs)
        if a.seed_only:
            phase='seed_pass_complete';publish('completed');return
        # Discover complete, dense geometry certificates without inheriting
        # old physical acceptance from a different actuator/MCC configuration.
        certificates = {}
        for cache in a.reference_caches:
            for path in cache.glob('planning/*/healthy_reference/reference.h5'):
                match = re.search(r'candidate_(\d+)', str(path))
                if not match: continue
                cid = int(match.group(1))
                summary = path.parent/'summary.json'
                if not summary.exists(): continue
                info = json.loads(summary.read_text())
                if not info.get('collection_admissible'): continue
                if info.get('hand_planner_sha256') != hashes['healthy_grasp_planner.py']: continue
                with h5py.File(path) as h:
                    valid = bool(h.attrs.get('healthy_reference_admissible'))
                    valid &= bool(np.all(h['grasp_keyframe_valid'][:]))
                if valid: certificates[cid] = path.resolve()
        seed_ids = {j['candidate_id'] for j in seed_manifest['jobs']}
        long_jobs = []
        for cid,path in sorted(certificates.items()):
            if cid in seed_ids or cid == 4: continue
            with h5py.File(path) as h: frames = len(h['palm_pose_object'])
            if frames < a.min_frames: continue
            long_jobs.append(base(cid,f'candidate_{cid:04d}_whole',reference=str(path),frames=frames,
                source_slice=[0,frames],kept_whole=True))
        phase='remaining_full_references';execute(long_jobs)
        children=[]
        for cid,original in sorted(originals.items()):
            if cid in seed_ids or cid == 4: continue
            whole = rows.get(f'candidate_{cid:04d}_whole')
            if whole and whole.get('selected'): continue
            if original.get('attempt_kind') != 'physical': continue
            summary_path = Path(original['source_output'])/'summary.json'
            if not summary_path.exists(): continue
            source = Path(json.loads(summary_path.read_text())['plan'])
            if not source.exists(): continue
            with h5py.File(source) as h: frames = len(h['palm_pose_object'])
            if frames < a.min_frames: continue
            for index,(begin,end) in enumerate(balanced_ranges(0,frames,a.max_child_frames,a.min_frames)):
                name=f'candidate_{cid:04d}_local{index:02d}'
                job=base(cid,name,frames=end-begin,source=str(source),source_slice=[begin,end],kept_whole=False)
                if cid in certificates:
                    ref=out/'plans'/f'{name}.h5'
                    if not ref.exists(): write_reference(certificates[cid],ref,begin,end,job['related_plan_group_id'])
                    job['reference']=str(ref)
                children.append(job)
        save(out/'original_child_jobs.json',dict(jobs=children))
        phase='original_children_first_pass';execute(children)
        # Only after the first pass, generate new geometries if the accepted
        # episode target has not been met. Do not repeatedly repair one route.
        for cid in range(max(originals)+1, max(originals)+1+a.max_new_parents):
            if len(accepted()) >= a.target: break
            resources();phase='new_reference_replenishment'
            folder=out/'new_parents'/f'candidate_{cid:04d}';folder.mkdir(parents=True,exist_ok=True)
            result_file=folder/'generation_result.json'
            if result_file.exists(): generated=json.loads(result_file.read_text())
            else:
                generated=attempt(candidate(cid,settings),argparse.Namespace(output=folder,stage='palm',
                    palm_source=None,palm_clearance_range_m=[.03,.04],max_wrist_speed_deg_s=60.))
                save(result_file,generated)
            if generated.get('status')!='palm_candidate':
                name=f'candidate_{cid:04d}_generation'
                if name not in rows:finish(dict(base(cid,name),status='palm_generation_failed',selected=False,
                    rejection_reasons=['palm_generation_failed']))
                continue
            source=Path(generated['palm_plan'])
            with h5py.File(source) as h: frames=len(h['palm_pose_object'])
            if frames<a.min_frames:continue
            execute([base(cid,f'candidate_{cid:04d}_local{i:02d}',frames=end-begin,source=str(source),
                source_slice=[begin,end],kept_whole=False)
                for i,(begin,end) in enumerate(balanced_ranges(0,frames,a.max_child_frames,a.min_frames))])
        phase='completed' if len(accepted())>=a.target else 'search_budget_exhausted'
        publish(phase)
    except BaseException as error:
        publish('interrupted',f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__': main()
