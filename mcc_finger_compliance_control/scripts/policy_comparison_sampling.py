"""Model-independent sampling weights; no validation/test episode can enter."""
from collections import Counter, defaultdict
import numpy as np


def balanced_weights(windows, episodes):
    lookup = {int(r['episode_id']): r for r in episodes}
    counts = Counter(eid for eid, _ in windows)
    cohorts = defaultdict(list)
    for eid in counts:
        row = lookup[eid]
        if row['split'] != 'train': raise ValueError(f'Nontraining episode {eid} in sampler')
        cohorts[row['cohort']].append(eid)
    if set(cohorts) != {'mustard', 'surface'}: raise ValueError('Expected old and new source pools')
    weights = np.asarray([1/(len(cohorts)*len(cohorts[lookup[eid]['cohort']])*counts[eid])
                          for eid, _ in windows], dtype=np.float64)
    return weights, dict(cohort_probability={k:1/len(cohorts) for k in cohorts},
                         episodes_per_cohort={k:len(v) for k,v in cohorts.items()},
                         total_weight=float(weights.sum()), train_windows=len(windows),
                         rule='equal cohort, equal episode, uniform within-episode window')
