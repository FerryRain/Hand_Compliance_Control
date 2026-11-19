"""Prepare the accepted regional teachers without changing their frozen split.

Reuses the DP0 inversion, motion96 export, and preissued-palm-command code.
All windows retain their physical timestamps and original episode boundaries.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from export_palm_dp import export
from invert_trajectories import invert_selected
from rebuild_commanded_palm_dp import commanded_delta, plan_indices, rebuild


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def selected_rows(collection: Path) -> list[dict]:
    payload = json.loads((collection / 'collection_manifest.json').read_text())
    rows = sorted((row for row in payload['rows'] if row.get('selected')),
                  key=lambda row: str(Path(row['selected_file']).resolve()))
    if not rows:
        raise ValueError('No selected teacher episodes')
    names: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        # Surface collections call the development split 'dev'. Preserve the
        # raw frozen label while exporting the trainer's equivalent 'val'.
        row['collection_split']=row['split']
        if row['split']=='dev':row['split']='val'
        path = Path(row['selected_file']).resolve()
        if path.stem in names:
            raise ValueError(f'Duplicate source name: {path.stem}')
        names.add(path.stem)
        split = row['split']
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'Invalid preassigned split: {split}')
        group_splits[row['related_plan_group_id']].add(split)
        if not (row['status'] == 'completed'
                and row['requested_frames'] == row['observed_frames']
                and row['observed_frames'] >= 800
                and row['stable_start'] and row['force_audit']['passed']
                and row['execution_contract']['passed']
                and row['strict_healthy_four_rate_full'] >= .95
                and min(row['strict_per_finger_full']) >= .95
                and row['longest_observed_four_loss_frames'] <= 10
                and row['force_events']['total_overforce_ms'] == 0
                and row['force_events']['nonfinite_samples'] == 0):
            raise ValueError(f'Selected episode failed its collection gate: {path}')
        if not path.is_file():
            raise FileNotFoundError(path)
    conflicts = {g: sorted(v) for g, v in group_splits.items() if len(v) > 1}
    if conflicts:
        raise ValueError(f'Parent groups cross splits: {conflicts}')
    if {r['split'] for r in rows} != {'train', 'val', 'test'}:
        raise ValueError('Need nonempty frozen train/val/test splits')
    return rows


def prepare(collection: Path, output: Path) -> dict:
    collection = collection.resolve()
    output = output.resolve()
    rows = selected_rows(collection)
    output.mkdir(parents=True, exist_ok=True)
    inverse = output / 'teachers_inverted.h5'
    base = output / 'teachers_motion96.h5'
    dp = output / 'teachers_commanded_palm.h5'
    split_path = output / 'episode_split.json'
    for path in (inverse, base, dp, split_path, output / 'data_audit.json'):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite prepared data: {path}')
    save(output / 'preparation_status.json', dict(state='checking_raw', episodes=len(rows)))
    sources = []
    for index, row in enumerate(rows):
        path = Path(row['selected_file']).resolve()
        with h5py.File(path, 'r') as raw:
            if (not raw.attrs.get('split_fingertip_collision', False)
                    or raw.attrs.get('collision_geometry') != 'true_fingertip_contact_v1_split'):
                raise ValueError(f'Unexpected collision model: {path}')
            if (raw.attrs['related_plan_group_id'] != row['related_plan_group_id']
                    or raw.attrs['exploration_split'] != row['collection_split']):
                raise ValueError(f'Raw metadata and manifest differ: {path}')
            if raw['q_hand'].shape != (row['observed_frames'], 1, 16):
                raise ValueError(f'Unexpected q shape: {path}')
            if not np.isclose(raw.attrs['control_dt'], .01, rtol=0, atol=1e-12):
                raise ValueError(f'Unexpected sample time: {path}')
            for name in ('q_hand', 'palm_command_pose_object', 'oracle_surface_normal_world'):
                if not np.isfinite(raw[name][:]).all():
                    raise ValueError(f'Nonfinite {name}: {path}')
        sources.append(dict(episode_id=index, source=str(path), sha256=digest(path),
                            related_plan_group_id=row['related_plan_group_id'],
                            split=row['split'], collection_split=row['collection_split'], family=row['family'], tube=bool(row['tube']),
                            frames=row['observed_frames']))
    save(output / 'source_manifest.json', dict(collection=str(collection), episodes=sources,
         collection_manifest_sha256=digest(collection / 'collection_manifest.json')))
    save(output / 'preparation_status.json', dict(state='inverting', episodes=len(rows)))
    print(f'[PREPARE] explicitly inverting {len(rows)} accepted teachers', flush=True)
    invert_selected([], [Path(r['source']) for r in sources], inverse)
    export(inverse, base, state_schema='contact_geometry_planner_motion',
           planner_waypoints=1, planner_step_frames=20, motion_feature_step_frames=5,
           action_field='q_hand')
    save(output / 'preparation_status.json', dict(state='rebuilding_preissued_commands'))
    command_stats = rebuild(base, inverse, [collection / 'selected'], dp)

    # Verify every frame, not just the first episode or an exported summary.
    maximum_command_error = 0.0
    with h5py.File(inverse, 'r+') as inv, h5py.File(dp, 'r+') as data:
        labels = [value.decode() if isinstance(value, bytes) else str(value)
                  for value in inv['source_trajectory'][:]]
        if labels != [r['source'] for r in sources]:
            raise ValueError('Inversion source ordering changed')
        cursor = 0
        for source in sources:
            with h5py.File(source['source']) as raw:
                count = source['frames']
                block = slice(cursor, cursor + count)
                for file in (inv, data):
                    if not np.array_equal(file['q_hand'][block], raw['q_hand'][:]):
                        raise ValueError(f"Measured q changed: {source['source']}")
                    if not np.all(file['episode_id'][block] == source['episode_id']):
                        raise ValueError('Episode boundary changed')
                mask = raw['true_fingertip_contact_valid'][:]
                if (not np.array_equal(inv['fingertip_contact'][block], mask)
                        or not np.array_equal(data['fingertip_contact_mask'][block], mask)):
                    raise ValueError(f"True-tip contact mask changed: {source['source']}")
                plan = raw['palm_command_pose_object'][:].reshape(-1, 7)
                schedule = raw['object_motion_schedule_step'][:, 0]
                lengths = np.unique(raw['object_segment_move_steps'][:, 0])
                if len(lengths) != 1:
                    raise ValueError('Changing motion length inside an episode')
                target_indices = plan_indices(schedule, 20, int(lengths[0]), len(plan),
                    raw.attrs.get('planner_motion_indexing') == 'endpoint_sample_count_minus_one_v1')
                expected = commanded_delta(inv['palm_pose_object'][block, 0], plan[target_indices])
                actual = data['planner_palm_delta_pose_palm'][block, 0, 0]
                error = float(np.max(np.abs(actual - expected)))
                maximum_command_error = max(maximum_command_error, error)
                if error > 1e-6:
                    raise ValueError(f'Preissued palm timing/frame mismatch: {error}')
                cursor += count
        if cursor != len(data['q_hand']):
            raise ValueError('Frame count mismatch')
        for name, field in (('source_plan_name', None),
                            ('source_parent_group', 'related_plan_group_id'),
                            ('source_frozen_split', 'split')):
            values = [Path(r['source']).stem if field is None else r[field] for r in sources]
            data.create_dataset(name, data=np.asarray(values, dtype=object),
                                dtype=h5py.string_dtype('utf-8'))
        for file in (inv, data):
            file.attrs['collection_manifest'] = str(collection / 'collection_manifest.json')
            file.attrs['variable_length_episodes'] = True
            file.attrs['episode_group_metadata'] = str(split_path)
            file.attrs['fixed_split_policy'] = 'preserve_collection_parent_geometry_split'
        for name, dataset in data.items():
            if dataset.dtype.kind in 'fc' and not np.isfinite(dataset[:]).all():
                raise ValueError(f'Nonfinite exported feature: {name}')
    save(split_path, dict(dp_h5=str(dp), action_field='q_hand', episodes=sources,
                         strategy='frozen_before_collection_parent_geometry'))
    counts = Counter(r['split'] for r in sources)
    groups = {split: len({r['related_plan_group_id'] for r in sources if r['split'] == split})
              for split in counts}
    families = {split: dict(Counter(r['family'] for r in sources if r['split'] == split))
                for split in counts}
    report = dict(episodes=len(sources), frames=cursor, split=dict(counts),
                  parent_groups=groups, families_by_split=families,
                  tube_episodes=sum(r['tube'] for r in sources),
                  clean_episodes=sum(not r['tube'] for r in sources),
                  measured_q_preserved=True, true_tip_labels_preserved=True,
                  episode_boundaries_preserved=True, parent_group_leakage=False,
                  preissued_command_max_abs_error=maximum_command_error,
                  commanded_palm_statistics=command_stats,
                  dp=str(dp), inverted=str(inverse), split_file=str(split_path))
    save(output / 'data_audit.json', report)
    save(output / 'preparation_status.json', dict(state='completed', **report))
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.collection, args.output)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        save(args.output / 'preparation_status.json',
             dict(state='failed', error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
