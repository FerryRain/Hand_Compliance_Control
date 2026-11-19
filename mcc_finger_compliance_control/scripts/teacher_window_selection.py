"""Explicit, provenance-bound window admission without trajectory stitching.

This module does not decide whether a teacher is good. It consumes a reviewed
manifest and preserves the original observations/labels at admitted anchors.
Normalization uses only admitted training histories and future labels.
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dp_dataset import Normalization, STATE_FIELDS_BY_SCHEMA


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def load_selection(path,dataset,split_path,episodes,obs_horizon,pred_horizon,stride):
    payload=json.loads(Path(path).read_text())
    if payload.get('training_approved') is not True:
        raise ValueError('Window candidates must be reviewed before training admission')
    for key,expected in [('obs_horizon',obs_horizon),('pred_horizon',pred_horizon),('stride',stride),
                         ('action_field','q_hand'),('action_representation','anchored_delta_q')]:
        if payload.get(key)!=expected:raise ValueError('Window selection mismatch: '+key)
    if Path(payload['dp_h5']).resolve()!=Path(dataset).resolve():raise ValueError('Wrong selected dataset')
    if payload['dataset_sha256']!=sha(dataset) or payload['split_sha256']!=sha(split_path):
        raise ValueError('Dataset or split changed after window review')
    split=json.loads(Path(split_path).read_text())
    expected={int(r['episode_id']):r for r in split['episodes']}
    selected={};objects={};groups={}
    for row in payload['episodes']:
        eid=int(row['episode_id'])
        if eid in selected or eid not in episodes or eid not in expected:
            raise ValueError('Duplicate or unknown selected episode')
        parent=expected[eid]
        for key in ('split','related_plan_group_id','object_id'):
            if row.get(key)!=parent.get(key) or row.get(key) is None:
                raise ValueError('Missing or conflicting grouped identity: '+key)
        objects.setdefault(row['object_id'],set()).add(row['split'])
        groups.setdefault(row['related_plan_group_id'],set()).add(row['split'])
        anchors=row['anchor_frames']
        if any(type(x) is not int or x<0 or x%stride for x in anchors):
            raise ValueError('Anchor must be a nonnegative raw frame on the sampling grid')
        current=np.asarray(anchors,dtype=int)//stride
        if len(set(anchors))!=len(anchors):raise ValueError('Duplicate selected anchor')
        n=len(episodes[eid][0])
        if np.any(current<obs_horizon-1) or np.any(current>=n-pred_horizon):
            raise ValueError('Incomplete history or future at selected anchor')
        selected[eid]=set(current.tolist())
    if set(selected)!=set(episodes) or set(expected)!=set(episodes):
        raise ValueError('Every episode, including zero-admission episodes, must be accounted for')
    if any(len(x)>1 for x in objects.values()) or any(len(x)>1 for x in groups.values()):
        raise ValueError('An object or related initial group crosses splits')
    return selected,dict(manifest_sha256=sha(path),dataset_sha256=payload['dataset_sha256'],
        split_sha256=payload['split_sha256'],selected_windows=sum(map(len,selected.values())),
        normalization_scope='Admitted train histories and labels only; overlapping histories retain sampling multiplicity',
        no_trajectory_stitching=True)


def selected_normalization(episodes,train_ids,selected,obs_horizon,pred_horizon,schema):
    dim=next(iter(episodes.values()))[0].shape[-1]
    total=np.zeros(dim);squares=np.zeros(dim);count=0
    bound=np.full((pred_horizon,16),1e-4)
    for eid in train_ids:
        state,q=episodes[eid]
        for current in sorted(selected[eid]):
            history=state[current-obs_horizon+1:current+1].astype(np.float64)
            delta=q[current+1:current+1+pred_horizon].astype(np.float64)-state[current,None,:16]
            if not (np.isfinite(history).all() and np.isfinite(delta).all()):
                raise ValueError('Nonfinite admitted teacher window')
            total+=history.sum(0);squares+=np.square(history).sum(0);count+=len(history)
            bound=np.maximum(bound,abs(delta))
    if not count:raise ValueError('No admitted training histories')
    mean=total/count;std=np.maximum(np.sqrt(np.maximum(squares/count-mean**2,0)),1e-5)
    cursor=0
    for name,size in STATE_FIELDS_BY_SCHEMA[schema]['palm']:
        if name=='fingertip_contact_mask':mean[cursor:cursor+size]=.5;std[cursor:cursor+size]=.5
        cursor+=size
    return Normalization(state_mean=mean.astype(np.float32),state_std=std.astype(np.float32),
        action_mean=np.zeros((pred_horizon,16),np.float32),action_std=bound.astype(np.float32))


def apply_selection(dataset,selected):
    before=len(dataset.windows)
    dataset.windows=[(eid,current) for eid,current in dataset.windows if current in selected[eid]]
    return dict(available_windows=before,admitted_windows=len(dataset.windows))


def action_range(episodes,ids,selected,pred_horizon,normalization):
    maximum=0.
    for eid in ids:
        state,q=episodes[eid]
        for current in selected[eid]:
            delta=q[current+1:current+1+pred_horizon]-state[current,None,:16]
            value=(delta-normalization.action_mean)/normalization.action_std
            if not np.isfinite(value).all():raise ValueError('Nonfinite normalized selected label')
            maximum=max(maximum,float(abs(value).max()))
    return maximum


def object_group_weights(windows, episodes):
    """Equal object / initial group / episode, then uniform admitted anchors.

    Corrects multiplicity of available demonstrations, not missing geometry or
    failed expert labels. Groups with no usable windows remain unrepresented.
    """
    lookup = {int(row['episode_id']): row for row in episodes}
    if len(lookup) != len(episodes) or not windows:
        raise ValueError('Unique episode identities and nonempty windows required')
    counts = Counter(eid for eid, _ in windows)
    by_object = defaultdict(lambda: defaultdict(list))
    group_objects = defaultdict(set)
    for eid in counts:
        row = lookup[eid]
        if row['split'] != 'train':
            raise ValueError('Validation/test episode in training sampler')
        obj, group = row['object_id'], row['related_plan_group_id']
        if not obj or not group:
            raise ValueError('Missing object or initial group identity')
        by_object[obj][group].append(eid)
        group_objects[group].add(obj)
    if any(len(objects) != 1 for objects in group_objects.values()):
        raise ValueError('An initial group belongs to multiple objects')
    mass = {}
    for obj, groups in by_object.items():
        for group, ids in groups.items():
            for eid in ids:
                mass[eid] = 1. / (len(by_object)*len(groups)*len(ids))
    weights = np.asarray([mass[eid]/counts[eid] for eid, _ in windows], dtype=np.float64)
    return weights, dict(rule='equal represented object, equal initial group, equal episode, uniform admitted anchor',
        object_probability={obj: 1./len(by_object) for obj in by_object},
        groups_per_object={obj: len(groups) for obj, groups in by_object.items()},
        episode_probability=mass, windows_per_episode=dict(counts), total_weight=float(weights.sum()),
        unrepresented_train_episodes=[eid for eid,row in lookup.items() if row['split']=='train' and eid not in counts],
        limitation='Cannot reconstruct missing state coverage or make failed actions valid teacher labels')
