"""Training-only reward adjustments and complete-coverage rollout scheduling."""
from collections import defaultdict, deque
import math
import numpy as np
import torch


def training_reward_adjustments(values, force, joint, cfg):
    """Keep v1 unchanged unless explicitly selected; never modifies an action.

    Above160N (at warning40N), add a logarithmic severity tail instead of
    treating all larger loads identically. The joint term uses the existing
    physical-range margin, not a teacher-q target or a fixed grasp posture.
    """
    version=cfg.get('reward_version','regional_v1')
    if version=='regional_v1':return values
    if version!='deployable_health_v2':raise ValueError(f'Unknown reward version: {version}')
    result=dict(values)
    normalized=((force-cfg['force_warn_n'])/(100-cfg['force_warn_n'])).clamp_min(0)
    result['force_tail']=-.8*torch.log1p((normalized-2).clamp_min(0))
    result['posture']=-.04*joint
    result['total']=sum(v for k,v in result.items() if k!='total')
    return result


def coverage_batches(rows,batch_size,seed,epoch):
    """Every training episode once per epoch, then pad the last partial batch.

    Neighboring lengths limit unused physics tails. Within length buckets,
    interleave parent groups; shuffle blocks so short paths are not always first.
    Do not invent mid-trajectory resets or randomly perturb initial conditions.
    """
    if batch_size<1 or not rows:raise ValueError('Nonempty data and positive batch size required')
    if any(r['split']!='train' for r in rows):raise ValueError('Training split only')
    if len({r['episode_id'] for r in rows})!=len(rows):raise ValueError('Duplicate source episodes')
    rng=np.random.default_rng(seed+epoch*104729)
    buckets=defaultdict(list)
    for row in rows:buckets[int(math.floor(math.log(max(1,row['frames']/800),1.5)))].append(row)
    batches=[]
    for key in sorted(buckets):
        groups=defaultdict(list)
        for row in buckets[key]:groups[row['related_plan_group_id']].append(row)
        queues={name:deque([group[i] for i in rng.permutation(len(group))]) for name,group in groups.items()}
        order=[]
        while queues:
            names=list(queues);rng.shuffle(names)
            for name in names:
                order.append(queues[name].popleft())
                if not queues[name]:del queues[name]
        for start in range(0,len(order),batch_size):
            block=order[start:start+batch_size]
            while len(block)<batch_size:block.append(order[int(rng.integers(len(order)))])
            batches.append([dict(row,branch=i) for i,row in enumerate(block)])
    rng.shuffle(batches)
    return batches
