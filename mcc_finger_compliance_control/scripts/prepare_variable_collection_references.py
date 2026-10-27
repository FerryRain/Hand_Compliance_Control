"""Prepare independently collectible, variable-length healthy references.

Intervals come from an offline contact/force audit. They identify useful
reference sections, not successful new rollouts. Every child must establish
its own stable physical initial state and pass collection acceptance again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np


def balanced_ranges(begin, end, maximum, minimum):
    count = (end - begin + maximum - 1) // maximum
    if count < 1 or (end - begin) // count < minimum:
        raise ValueError("Interval cannot meet both segment length bounds")
    width, extra = divmod(end - begin, count)
    result = []
    for index in range(count):
        stop = begin + width + int(index < extra)
        result.append((begin, stop))
        begin = stop
    return result


def write_reference(source, output, begin, end, family):
    """Copy exact reference samples, including all dense health witnesses."""
    if output.exists():
        raise FileExistsError(output)
    with h5py.File(source) as original:
        frames = len(original['palm_pose_object'])
        if not 0 <= begin < end <= frames:
            raise ValueError("Reference interval outside source")
        if not bool(original.attrs.get('healthy_reference_admissible', False)):
            raise ValueError("Source has not passed healthy-reference checks")
        indices = original['grasp_keyframe_frame_index'][:, 0]
        if not np.array_equal(indices, np.arange(frames)):
            raise ValueError("Expected dense certified keyframes; sparse ones need explicit boundary refinement")
        for name in ('grasp_keyframe_valid', 'grasp_keyframe_true_tip_geometry_valid'):
            if not np.all(original[name][begin:end]):
                raise ValueError(f"Invalid reference geometry: {name}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, 'w') as target:
            for key, value in original.attrs.items():
                target.attrs[key] = value
            for name, dataset in original.items():
                if not isinstance(dataset, h5py.Dataset):
                    raise ValueError(f"Unsupported reference group: {name}")
                if dataset.ndim and dataset.shape[0] == frames:
                    values = dataset[begin:end]
                    if name in ('grasp_keyframe_frame_index', 'record_step'):
                        values = values - begin
                    target.create_dataset(name, data=values, dtype=dataset.dtype)
                    for key, value in dataset.attrs.items():
                        target[name].attrs[key] = value
                else:
                    original.copy(name, target)
            # A child can itself be split again after an independent rollout.
            # Preserve its original frame provenance rather than creating the
            # same dataset twice or replacing original indices with local ones.
            if 'subtrajectory_source_frame' not in target:
                target.create_dataset('subtrajectory_source_frame', data=np.arange(begin, end))
            target.attrs.update(
                subtrajectory_reference=True,
                subtrajectory_source=str(source.resolve()),
                subtrajectory_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                subtrajectory_source_slice=np.asarray([begin, end], dtype=np.int64),
                related_plan_group_id=family,
                subtrajectory_timing='Exact source samples and control_dt; no time stretching or compression',
                subtrajectory_physical_initialization='Required independently; not certified by reference slicing',
                subtrajectory_collection_accepted=False)
            for name in ('palm_pose_object', 'palm_twist_object', 'q_hand', 'qvel',
                         'grasp_keyframe_q', 'grasp_keyframe_non_tip_distance',
                         'grasp_keyframe_local_motion_capacity_m'):
                np.testing.assert_array_equal(target[name][:], original[name][begin:end])
            np.testing.assert_array_equal(target['grasp_keyframe_frame_index'][:, 0], np.arange(end-begin))
            np.testing.assert_array_equal(target['record_step'][:, 0], np.arange(end-begin))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--segment-candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=2000)
    parser.add_argument('--min-frames', type=int, default=500)
    args = parser.parse_args()
    if not 2 <= args.min_frames <= args.max_frames:
        parser.error('Expected 2 <= min-frames <= max-frames')
    output = args.output.resolve()
    if output.exists():
        parser.error('Use a fresh output directory')
    output.mkdir(parents=True)
    rows = [r for r in json.loads(args.comparison.read_text())['rows']
            if r['variant'] == '10N handover v2']
    candidates = json.loads(args.segment_candidates.read_text())
    scripts = Path(__file__).resolve().parent
    jobs, excluded = [], []
    for row in rows:
        cid = row['candidate_id']
        summary = json.loads((Path(row['output'])/'summary.json').read_text())
        if summary.get('actual_motion_start_step') is None:
            excluded.append(dict(candidate_id=cid, reason='Initialization failed; no usable motion interval'))
            continue
        source = Path(summary['plan'])
        with h5py.File(source) as handle:
            family = str(handle.attrs.get('related_plan_group_id',
                         handle.attrs.get('healthy_reference_source', str(source.resolve()))))
            dt = float(handle.attrs['control_dt'])
        accepted_whole = (bool(row['stable_start']) and row['status'] == 'completed'
            and row['strict_healthy_four_rate_full'] >= .95
            and min(row['strict_per_finger_full']) >= .95
            and row['longest_observed_four_loss_frames'] <= 10
            and row['peak_all_parts_n'] <= 100.)
        if accepted_whole:
            ranges = [(0, int(row['requested_frames']))]
        else:
            ranges = []
            for segment in candidates['segments']:
                if segment['candidate_id'] == cid:
                    ranges.extend(balanced_ranges(*segment['source_record_slice'],
                                                  args.max_frames, args.min_frames))
        for index, (begin, end) in enumerate(ranges):
            name = f'candidate_{cid:04d}_s{index:02d}'
            reference = output/'plans'/f'{name}.h5'
            write_reference(source, reference, begin, end, family)
            folder = output/'rollouts'/name
            seed = int(row['seed']) + 1000000 + index
            command = [sys.executable, '-u', str(scripts/'collect_leap_force_matched.py'),
                '--leap-profile', 'peak', '--leap-kp', '35', '--leap-kd', '.2',
                '--mcc-force-targets', '10', '10', '10', '10', '--recovery-handover',
                '--object-substep-interpolation', '--plan', str(reference), '--output', str(folder),
                '--frames', str(end-begin), '--stable-initialization',
                '--initialization-extra-steps', '1200', '--force-profile', 'regulated',
                '--force-audit', '--force-parts', '--substep-force-safety',
                '--overforce-response', 'record', '--seed', str(seed)]
            if row['tube']:
                command.extend(['--execution-randomization', '--execution-clean-env-fraction', '0'])
            jobs.append(dict(name=name, candidate_id=cid, variant='independent_subtrajectory',
                source_reference=str(source), source_slice=[begin, end], frames=end-begin,
                duration_s=(end-begin)*dt, preserved_full_qualified_reference=accepted_whole,
                related_plan_group_id=family, seed=seed, tube=row['tube'],
                reference=str(reference), output=str(folder), command=command,
                status='prepared_not_collected'))
    manifest = dict(jobs=jobs, excluded=excluded, summary=dict(references=len(jobs),
        families=len({j['candidate_id'] for j in jobs}),
        kept_whole=sum(j['preserved_full_qualified_reference'] for j in jobs),
        split_children=sum(not j['preserved_full_qualified_reference'] for j in jobs)),
        initialization='New independent stable physical start for every child; no snapshot or source live q reuse',
        acceptance=dict(strict_four_min=.95, strict_per_finger_min=.95,
                        longest_four_loss_max_frames=10, max_force_n=100.),
        force_response='Record and continue; over100N still reported for data-quality acceptance',
        split_rule='All children and Tube variants of a base reference remain in one train/dev/test group',
        qualification='Reference preparation only; actual new initialization and complete child rollouts must pass again',
        startup_caveat='Exact path samples preserve source velocities. Newly cut starts may have nonzero motion; dynamic startup has not yet been validated. No unverified smoothing is applied.')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest['summary'], indent=2))


if __name__ == '__main__':
    main()
