"""Repeat the frozen suite with the wrist load supervisor as the only controller change."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from unittest.mock import patch

import torch

from . import sweep_experiment as runtime
from . import sweep_verify
from .experiment import digest,save_json
from .load_guard import GuardedWrist
from .sweep_paced import timer_resolution
from .sweep_score import score
from .sweep_suite import jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('freeze','run','score','verify'))
    p.add_argument('--root',type=Path,default=runtime.OUT)
    p.add_argument('--group')
    a=p.parse_args()
    torch.set_num_threads(1)
    root=a.root/'force-priority'
    root.mkdir(exist_ok=True,parents=True)
    registered=json.loads((a.root/'protocol.json').read_bytes())
    hashes=runtime.hashes()
    if hashes!=registered['wrist_code']:
        raise ValueError('Unsupervised comparison source changed')
    extended={**hashes,'load_guard.py':digest(Path(__file__).with_name('load_guard.py')),
              'sweep_guard_trials.py':digest(Path(__file__))}
    spec=dict(source=extended,parent_protocol=digest(a.root/'protocol.json'),jobs=jobs(),
              decision='Registered after observing a 693 N palm transient in the original recovery.',
              detector='regional peak >=16 N or palm >=6 N; clear after 5 ticks with peak<5 N and palm<2 N',
              response='freeze tangent and rotation, retract radially at up to 0.16 m/s, keep DP/MCC running')
    protocol=root/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_bytes())!=spec:
        raise ValueError('Wrist guard protocol changed')
    if not protocol.exists():
        save_json(protocol,spec)
    if a.action=='freeze':
        return
    if a.action=='verify':
        with patch.object(sweep_verify,'SweepWrist',GuardedWrist):
            sweep_verify.replay(root)
        return
    # Inject the alternative wrist class into an isolated trial process. Keep
    # the original, fingerprinted integration loop intact for the paired test.
    with patch.object(runtime,'SweepWrist',GuardedWrist),patch.object(runtime,'hashes',return_value=extended):
        for r in spec['jobs']:
            if a.group and r['group']!=a.group:
                continue
            folder=root/r['series']/r['obj']/r['start']/r['arm']
            if a.action=='run':
                with (timer_resolution() if r['paced'] else nullcontext()):
                    record=runtime.run(folder,r['obj'],r['start'],r['arm'],r['seconds'],r['paced'],r['seed'])
                expected=registered['fixtures'][r['obj']+'/'+r['start']]
                if record['initial_file_sha256']!=expected['initial.npz'] or record['model_sha256']!=expected['model.mjb']:
                    raise ValueError('Guarded run used a different physical start')
            else:
                score(folder,folder,r['obj'],r['start'],[s for s in (60.,120.,240.,600.,1129.32) if s<=r['seconds']])


if __name__=='__main__':
    main()
