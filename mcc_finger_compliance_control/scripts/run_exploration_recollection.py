"""Collect fresh Mustard exploration teachers with the audited baseline.

CPU planning overlaps independent single-environment CUDA collectors. Failed
rollouts remain available for diagnostics and never enter selected/. There is
no DP/RL training in this coordinator.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
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

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
DATA = SCRIPTS.parent / 'data'
CONTROL_FILES = (
    'collect_long_oracle_probe.py', 'collect_trajectories.py',
    'surface_mcc_finger.py', 'collection_force_regulation.py',
    'collection_force_experiment.py', 'true_fingertip_contact.py',
    'independent_contact_force_audit.py', 'plan_exploration_dataset.py',
    'optimize_contact_plan.py', 'true_tip_waypoint_optimizer.py',
    'generate_manifold_palm_plan.py', 'replan_grasp_221.py',
    'bundle_manifold_plans.py', 'collection_force_events.py',
    'analyze_grasp_clearance.py', 'run_collection_force_suite.py',
    'run_exploration_recollection.py',
    'healthy_grasp_planner.py','replan_healthy_reference.py','collect_healthy_reference.py',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def label_split(group, seed):
    # Geometry, paired speed variants and all tube variants share a split.
    value = np.random.default_rng(seed + 700001 + group).random()
    return 'train' if value < .70 else 'val' if value < .85 else 'test'


def qualifies(row):
    return bool(row['status'] == 'completed'
        and row['stable_start'] and row['force_audit']['passed']
        and row['strict_healthy_four_rate_full'] >= .98
        and min(row['strict_per_finger_full']) >= .98
        and row['longest_observed_four_loss_frames'] <= 10
        and row['force_events']['total_overforce_ms'] == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=DATA / 'exploration_recollection_20261015')
    parser.add_argument('--target', type=int, default=221)
    parser.add_argument('--planning-batch', type=int, default=32)
    parser.add_argument('--max-candidates', type=int, default=1024)
    parser.add_argument('--planning-workers', type=int, default=4)
    parser.add_argument('--collect-workers', type=int, default=3)
    parser.add_argument('--min-free-gb', type=float, default=15.)
    parser.add_argument('--seed', type=int, default=20261015)
    args = parser.parse_args()
    if min(args.target, args.planning_batch, args.max_candidates,
           args.planning_workers, args.collect_workers) < 1:
        parser.error('positive counts required')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'coordinator.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1', MPLCONFIGDIR='/tmp/matplotlib',
        WARP_CACHE_PATH='/tmp/warp', TQDM_DISABLE='1')
    hashes = {f: digest(SCRIPTS / f) for f in CONTROL_FILES}
    protocol = dict(config={k: str(v) if isinstance(v, Path) else v
                            for k, v in vars(args).items()},
        code_sha256=hashes, object='ycb_mustard',
        controls=dict(arm_kp=8000, arm_kd=160,
            force_profile='regulated', force_targets_n=[6, 6, 6, 8],
            outward_budget_m=.006, outward_synergy_step_rad=.025,
            control_dt_s=.01, physics_dt_s=.001,
            preparation_frames=800, settling_frames=400),
        motion=dict(speeds_m_s=[.03, .05], clearance_range_m=[.03, .04],
            posture_gain=.18, hard_non_tip_clearance_m=.002,
            preferred_non_tip_clearance_m=.005, fixed_frame_padding=False),
        tube=dict(clean_probability=.25, norm_range_rad=[.003, .020],
            norm_semantics='16D L2, log uniform; not per-joint bounds',
            time_constant_s=.35, ramp_control_frames=50,
            assignment='independent seeded draw per geometry; paired speeds share assignment'),
        overforce_response='record', force_event_threshold_n=100,
        acceptance=dict(strict_four=.98, strict_per_finger=.98,
            longest_four_loss_frames=10, complete=True, over100_ms=0,
            independent_force_audit=True, calibrated_start=True),
        split='geometry-group 70/15/15 preassigned; paired speeds remain together; bootstrap is diagnostic only',
        scope='Fresh cross-region teacher collection, no training; rejected raw retained')
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('Protocol/code changed; use a fresh output directory')
    save(protocol_path, protocol)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in CONTROL_FILES:
        shutil.copy2(SCRIPTS / name, snapshot / name)
    selected = out / 'selected'
    selected.mkdir(exist_ok=True)
    rows = [json.loads(p.read_text()) for p in (out / 'rollouts').glob('*/selection.json')]
    done = {r['case'] for r in rows}
    active = {}
    started = time.monotonic()

    def accepted():
        return sum(r['selected'] for r in rows if not r['diagnostic_only'])

    def status(phase, **extra):
        state = dict(phase=phase, updated_local=time.strftime('%Y-%m-%d %H:%M:%S'),
            coordinator_pid=os.getpid(), target=args.target,
            accepted=accepted(), measured=len(rows),
            diagnostic_rollouts=sum(r['diagnostic_only'] for r in rows),
            active=list(active.values()), elapsed_s=time.monotonic()-started, **extra)
        save(out / 'status.json', state)
        save(out / 'collection_manifest.json', dict(protocol='protocol.json', rows=rows))

    def check_resources():
        for name, expected in hashes.items():
            if digest(SCRIPTS / name) != expected:
                raise RuntimeError(f'Live code changed: {name}; collection paused')
        if shutil.disk_usage(out).free < args.min_free_gb * 1024**3:
            raise RuntimeError('Free disk below configured floor; collection paused')

    def collect(job):
        from bundle_manifold_plans import bundle
        from run_collection_force_suite import analyze_job
        check_resources()
        folder = out / 'rollouts' / job['case']
        folder.mkdir(parents=True, exist_ok=True)
        plan = folder / 'plan.h5'
        source = Path(job['source'])
        with h5py.File(source) as h:
            frames = len(h['palm_pose_object'])
            batched = h['grasp_keyframe_q'].ndim == 3
        if not plan.exists():
            if batched:
                shutil.copy2(source, plan)
            else:
                bundle([source], plan)
        with h5py.File(plan) as h:
            if h['grasp_keyframe_q'].shape[1:] != (1, 16):
                raise ValueError(f'Invalid single-environment plan: {plan}')
            healthy=bool(h.attrs.get('healthy_reference_v1',False))
            if healthy and not h.attrs.get('healthy_reference_admissible',False):
                raise ValueError('Rejected healthy reference is not collectable')
        entry='collect_healthy_reference.py' if healthy else 'collect_long_oracle_probe.py'
        command = [sys.executable, str(SCRIPTS / entry),
            '--plan', str(plan), '--output', str(folder), '--frames', str(frames),
            '--force-profile', 'regulated', '--arm-stiffness-scale', '1',
            '--mcc-authority', 'baseline', '--overforce-response', 'record',
            '--force-audit', '--force-parts', '--substep-force-safety',
            '--seed', str(job['seed'])]
        if job['tube']:
            command += ['--execution-randomization', '--execution-clean-env-fraction', '0']
        save(folder / 'job.json', dict(job, command=command,
             source_sha256=digest(source), frames=frames))
        if not (folder / 'summary.json').exists():
            with (folder / 'run.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT)
            if not (folder / 'summary.json').exists():
                raise RuntimeError(f'{job["case"]}: collector exit {result.returncode}; see run.log')
        record = analyze_job(dict(job, arm='baseline', plan=str(plan),
            frames=frames, output=str(folder)))
        ok = qualifies(record)
        record.update(selected=bool(ok and not job['diagnostic_only']),
            quality_gate_passed=ok, diagnostic_only=job['diagnostic_only'],
            tube=job['tube'], split=job['split'], source_plan=str(source))
        if record['selected']:
            destination = selected / f'{job["case"]}.h5'
            if not destination.exists():
                shutil.copy2(folder / 'collected.h5', destination)
            with h5py.File(destination, 'r+') as h:
                h.attrs.update(exploration_geometry_group=str(job['geometry_group']),
                    exploration_split=job['split'], source_plan=str(source),
                    exploration_family=job['family'], recollection_seed=job['seed'])
        save(folder / 'selection.json', record)
        return record

    def finish_ready(block=False):
        if not active:
            return
        ready, _ = wait(active, timeout=10 if block else 0,
                        return_when=FIRST_COMPLETED)
        for future in ready:
            active.pop(future)
            row = future.result()
            rows.append(row)
            done.add(row['case'])
            print(f'[RECOLLECT] {row["case"]}: strict={row["strict_healthy_four_rate_full"]:.3%} '
                f'found={row["found_four_rate_full"]:.3%} '
                f'over100={row["force_events"]["total_overforce_ms"]}ms '
                f'selected={row["selected"]}; accepted={accepted()}/{args.target}', flush=True)

    planner = None
    try:
        with ThreadPoolExecutor(max_workers=args.collect_workers) as pool:
            # Repeat a known full orbit, with and without the existing tube,
            # before interpreting fresh planning failures. Never counts toward 221.
            reference = DATA / 'exploration_20261015/orbit_clearance_repair/teacher_hard2_preferred5.h5'
            for variant in range(3):
                name = f'bootstrap_orbit_{variant}'
                if name in done:
                    continue
                while len(active) >= args.collect_workers:
                    finish_ready(block=True)
                job = dict(case=name, source=str(reference), family='oblique_orbit',
                    geometry_group='bootstrap_orbit', tube=variant != 0,
                    seed=args.seed+200000+variant, split='diagnostic', diagnostic_only=True)
                active[pool.submit(collect, job)] = name
            for start in range(0, args.max_candidates, args.planning_batch):
                if accepted() >= args.target:
                    break
                check_resources()
                count = min(args.planning_batch, args.max_candidates-start)
                planning = out / 'planning' / f'round_{start:04d}'
                planning.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(SCRIPTS / 'plan_exploration_dataset.py'),
                    '--output', str(planning), '--start', str(start),
                    '--candidates', str(count), '--workers', str(args.planning_workers),
                    '--stage', 'teacher', '--seed', str(args.seed),
                    '--speeds-m-s', '.03', '.05', '--palm-clearance-range-m', '.03', '.04']
                save(planning / 'launch.json', dict(command=command))
                scheduled = set()
                with (planning / 'coordinator.log').open('a') as log:
                    planner = subprocess.Popen(command, cwd=ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT)
                    while True:
                        finish_ready()
                        for path in sorted(planning.glob('candidate_*/result.json')):
                            row = json.loads(path.read_text())
                            cid = row['candidate_id']
                            name = f'candidate_{cid:04d}'
                            if name in done or name in scheduled or not row.get('collection_candidate'):
                                continue
                            if len(active) >= args.collect_workers:
                                break
                            if accepted()+len(active) >= args.target:
                                break
                            group = row['geometry_group']
                            tube = np.random.default_rng(args.seed+300001+group).random() >= .25
                            job = dict(case=name, source=row['collection_plan'],
                                family=row['family'], geometry_group=group,
                                seed=args.seed+100000+cid, tube=bool(tube),
                                split=label_split(group, args.seed), diagnostic_only=False)
                            active[pool.submit(collect, job)] = name
                            scheduled.add(name)
                            print(f'[RECOLLECT] start {name} tube={tube}', flush=True)
                        finished_planning = planner.poll() is not None
                        admitted = {f'candidate_{json.loads(p.read_text())["candidate_id"]:04d}'
                            for p in planning.glob('candidate_*/result.json')
                            if json.loads(p.read_text()).get('collection_candidate')}
                        status('planning_and_collecting', planning_start=start,
                            planning_finished=finished_planning,
                            planning_results=len(list(planning.glob('candidate_*/result.json'))),
                            collection_candidates=len(admitted), free_gb=shutil.disk_usage(out).free/1024**3)
                        if accepted() >= args.target:
                            if not finished_planning:
                                planner.wait()
                            break
                        if finished_planning and admitted <= (done | scheduled):
                            break
                        time.sleep(5)
                    if planner.wait() != 0:
                        raise RuntimeError(f'Planning failed: {planning}/coordinator.log')
                    planner = None
                while active:
                    finish_ready(block=True)
                    status('collecting', planning_start=start)
                if start >= args.planning_batch and not accepted():
                    raise RuntimeError('No qualified data after two fresh planning batches; inspect teacher feasibility before generating more')
            status('complete' if accepted() >= args.target else 'candidate_budget_exhausted')
    except BaseException as error:
        if planner is not None and planner.poll() is None:
            planner.terminate()
            planner.wait()
        status('error', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
