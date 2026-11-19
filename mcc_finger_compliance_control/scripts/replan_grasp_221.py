"""Replan all original221 path configurations with new clearance and grasp rules.

Produces teacher collection candidates, never relabels old measured actions.
Continuous geometry checks reproduce the collector's valid-knot smoothstep.
Physical collection/strict+found scoring must follow before DP training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np

from generate_manifold_palm_plan import sample_palm_clearance

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
DATA = ROOT / 'mcc_finger_compliance_control/data'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_source(episode, args):
    source = Path(episode['source_plan'])
    if not source.is_absolute():
        source = ROOT / source
    with h5py.File(source) as handle:
        candidate = json.loads(handle.attrs['sampling_candidate_json'])
        attrs = handle.attrs
        if attrs['planner_path_mode'] != 'minimum_enclosing_ellipse':
            raise ValueError(f'unsupported source path mode: {source}')
        row = dict(episode=int(episode['episode']), source_plan=str(source.resolve()),
                   source_sha256=sha256(source), family=candidate['family'], candidate=candidate,
                   object_id=str(attrs['object_id']), source_frames=len(handle['palm_pose_object']),
                   object_initial_pose=np.asarray(attrs['planner_object_initial_pose_world']).tolist(),
                   global_plane_mode=str(attrs.get('planner_global_ellipse_plane_mode', 'calibrated')))
    row['clearance_seed'] = args.seed + row['episode']
    row['sampled_clearance_m'] = sample_palm_clearance(args.palm_clearance_range_m, row['clearance_seed'])
    return row


def commands(row, args, raw, optimized):
    c = row['candidate']
    generate = [sys.executable, str(SCRIPTS / 'generate_manifold_palm_plan.py'),
                '--output', str(raw), '--object-id', row['object_id'],
                '--frames', str(args.frames), '--path-mode', 'minimum_enclosing_ellipse',
                '--angle-deg', str(c['angle_deg']), '--direction', str(c['direction']),
                '--ellipse-section-fraction', str(c['section_fraction']),
                '--ellipse-plane-tilt-deg', str(c['tilt_deg']),
                '--ellipse-azimuth-deg', str(c['azimuth_deg']),
                '--ellipse-meridian-deg', str(c['meridian_deg']),
                '--ellipse-arc-region', c['arc_region'],
                '--ellipse-arc-center-offset-deg', str(c['arc_center_offset_deg']),
                '--global-ellipse-plane-mode', row['global_plane_mode'],
                '--palm-tangent-sign', '-1', '--object-rotation-mode', 'uniform_so3',
                '--seed', str(c['object_seed']), '--time-parameterization', 'blend',
                '--palm-yaw-offset-deg', str(c['palm_yaw_offset_deg']),
                '--palm-yaw-end-offset-deg', str(c['palm_yaw_end_offset_deg']),
                '--palm-clearance-range-m', *map(str, args.palm_clearance_range_m),
                '--palm-clearance-seed', str(row['clearance_seed'])]
    optimize = [sys.executable, str(SCRIPTS / 'optimize_contact_plan.py'),
                '--input', str(raw), '--output', str(optimized),
                '--object-id', row['object_id'], '--true-tip-waypoints', '--keyframes-only',
                '--reference-bank', str(args.reference_bank.resolve()),
                '--knot-stride', str(args.knot_stride), '--max-posture-deviation-rad', '.90',
                '--ik-posture-gain', '.18', '--non-tip-clearance-m', '.002',
                '--max-cross-finger-penetration-m', '.0001', '--tip-top-weight', '.05',
                '--tip-top-preferred-fraction', '.75', '--no-extract-feasible-segment']
    return generate, optimize


def continuous_geometry(path, output):
    from analyze_grasp_clearance import smooth_keyframe_reference
    from object_catalog import load_object_config
    from true_tip_waypoint_optimizer import SplitHullGeometry, TrueTipWaypointController
    with h5py.File(path) as handle:
        palm = np.asarray(handle['palm_pose_object'])[:, 0]
        frames = np.asarray(handle['grasp_keyframe_frame_index']).reshape(-1)
        knots = np.asarray(handle['grasp_keyframe_q']).reshape(-1, 16)
        valid = np.asarray(handle['grasp_keyframe_valid']).reshape(-1).astype(bool)
        object_id = str(handle.attrs['object_id'])
        scale = float(handle.attrs.get('object_scale', 1.))
        margin = float(handle.attrs.get('non_tip_clearance_m', .002))
        self_limit = float(handle.attrs.get('max_cross_finger_penetration_m', .0001))
    if valid.sum() < 2:
        return dict(passed=False, checked_frames=0, reason='fewer_than_two_valid_knots')
    geometry = SplitHullGeometry(load_object_config(object_id), scale)
    checker = TrueTipWaypointController.__new__(TrueTipWaypointController)
    checker.non_tip_clearance_m = margin
    checker.max_cross_finger_penetration_m = self_limit
    q = smooth_keyframe_reference(frames, knots, valid, len(palm))
    mask, distances, pad, self_depth, cut_caps = [], [], [], [], []
    for frame, pose in enumerate(palm):
        g = geometry.evaluate(q[frame], pose, self_collision=True, non_tip_clearance=True)
        mask.append(checker.geometry_valid(g))
        distances.append(g['non_tip_distance'])
        pad.append(g['distance'])
        self_depth.append(g['self_penetration'])
        cut_caps.append(g['cut_cap'])
    mask = np.asarray(mask, dtype=bool)
    distances = np.asarray(distances)
    np.savez_compressed(output, valid=mask, non_tip_distance=distances,
                        pad_distance=np.asarray(pad), cross_finger_penetration=self_depth,
                        non_tip_clearance_m=margin,cut_cap=np.asarray(cut_caps))
    return dict(passed=bool(mask.all()), checked_frames=len(mask),
                valid_fraction=float(mask.mean()), first_invalid_frame=(int(np.flatnonzero(~mask)[0])
                    if not mask.all() else None), non_tip_min_m=float(distances.min()),
                cross_finger_max_m=float(np.max(self_depth)),
                non_tip_clearance_m=margin,
                scope='serialized palm and collector smoothstep posture prior; not dynamic execution')


def attempt(row, args):
    folder = args.output / f"ep{row['episode']:04d}"
    folder.mkdir(exist_ok=True)
    result_path = folder / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    raw, optimized = folder / 'palm.h5', folder / 'optimized.h5'
    generate, optimize = commands(row, args, raw, optimized)
    result = dict(row, passed=False, plan=str(optimized.resolve()),
                  generation_command=generate, optimization_command=optimize)
    started = time.monotonic()
    try:
        with (folder / 'planning.log').open('w') as log:
            for stage, command in [('palm_generation', generate), ('hand_optimization', optimize)]:
                print(f'[GRASP221] ep{row["episode"]:03d} {stage}', flush=True)
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if process.returncode:
                    result.update(status=stage+'_error', return_code=process.returncode)
                    return result
                if stage == 'palm_generation':
                    with h5py.File(raw, 'r+') as handle:
                        actual = np.asarray(handle.attrs['planner_object_initial_pose_world'])
                        if not np.allclose(actual, row['object_initial_pose'], atol=1e-12, rtol=0):
                            raise ValueError('source object initial pose changed')
                        handle.attrs['sampling_family'] = row['family']
                        handle.attrs['sampling_candidate_json'] = json.dumps(row['candidate'])
                        handle.attrs['replan_source_episode'] = row['episode']
                        handle.attrs['replan_source_plan'] = row['source_plan']
        with h5py.File(optimized) as handle:
            valid = np.asarray(handle['grasp_keyframe_valid']).reshape(-1).astype(bool)
            result.update(valid_keyframes=int(valid.sum()), total_keyframes=len(valid),
                          measured_palm_min_m=float(handle['planner_palm_outline_min_clearance_object'][:].min()))
        result['continuous_geometry'] = continuous_geometry(optimized, folder / 'geometry.npz')
        result['passed'] = bool(valid.all() and result['continuous_geometry']['passed'])
        result['status'] = 'geometry_candidate' if result['passed'] else 'geometry_rejected'
    except Exception as error:
        result.update(status='error', error=f'{type(error).__name__}: {error}')
    finally:
        result['elapsed_s'] = time.monotonic() - started
        write_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-summary', type=Path,
        default=DATA / 'true_tip_v2_20261009/rescreen221_final/summary.json')
    parser.add_argument('--reference-bank', type=Path,
        default=DATA / 'dp0_true_tip_1500/long_chains/training_only_healthy_seed_bank.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--frames', type=int, default=1500)
    parser.add_argument('--knot-stride', type=int, default=100)
    parser.add_argument('--palm-clearance-range-m', type=float, nargs=2, default=(.030, .040))
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--episodes', type=int, nargs='+')
    parser.add_argument('--dry-run', action='store_true', help='Validate all inputs and save planned commands only')
    args = parser.parse_args()
    if args.workers < 1 or args.frames < 160 or args.knot_stride < 1:
        parser.error('positive workers/stride and >=160 frames required')
    sample_palm_clearance(args.palm_clearance_range_m, args.seed)
    episodes = json.loads(args.source_summary.read_text())['episodes']
    if {int(x['episode']) for x in episodes} != set(range(221)) or len(episodes) != 221:
        raise ValueError('source must contain exactly original221 episode identities')
    if args.episodes is not None:
        if not args.episodes or not set(args.episodes) <= set(range(221)):
            parser.error('episodes must belong to original221')
        episodes = [x for x in episodes if int(x['episode']) in args.episodes]
    with np.load(args.reference_bank) as bank:
        if bank['q_hand'].shape[1:] != (16,) or bank['palm_pose_object'].shape != (len(bank['q_hand']), 7):
            raise ValueError('invalid reference bank dimensions')
    rows = [read_source(episode, args) for episode in sorted(episodes, key=lambda x:x['episode'])]
    protocol = dict(source_summary_sha256=sha256(args.source_summary),
        reference_bank_sha256=sha256(args.reference_bank), frames=args.frames, knot_stride=args.knot_stride,
        seed=args.seed, palm_clearance_range_m=list(args.palm_clearance_range_m),
        source_rows=rows, posture_gain=.18, non_tip_clearance_m=.002,
        scripts={name:sha256(SCRIPTS/name) for name in ['replan_grasp_221.py',
            'generate_manifold_palm_plan.py','optimize_contact_plan.py','true_tip_waypoint_optimizer.py']},
        stage='palm regeneration, hand refinement, every-frame geometry; physical collection follows')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('resume protocol changed; choose a fresh output directory')
    write_json(protocol_path, protocol)
    if args.dry_run:
        write_json(args.output / 'commands.json', [dict(episode=r['episode'],
            commands=commands(r,args,args.output/f'ep{r["episode"]:04d}/palm.h5',
                args.output/f'ep{r["episode"]:04d}/optimized.h5')) for r in rows])
        print(f'[GRASP221] preflight OK: {len(rows)} source trajectories; no planner/collector/training launched')
        return
    completed = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(attempt, row, args) for row in rows]
        for future in as_completed(futures):
            result = future.result(); completed.append(result)
            completed.sort(key=lambda r:r['episode'])
            summary = dict(requested=len(rows), completed=len(completed),
                qualified=sum(x['passed'] for x in completed), elapsed_s=time.monotonic()-started,
                rows=completed, qualification='all sparse knots and every-frame geometric acceptance; not physical data acceptance')
            write_json(args.output / 'progress.json', summary)
            (args.output / 'passing_plans.txt').write_text(''.join(x['plan']+'\n' for x in completed if x['passed']))
            print(f'[GRASP221] {len(completed)}/{len(rows)} qualified={summary["qualified"]}; '
                  f'ep{result["episode"]:03d} {result["status"]}', flush=True)
    write_json(args.output / 'summary.json', summary)


if __name__ == '__main__':
    main()
