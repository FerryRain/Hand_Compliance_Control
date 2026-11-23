"""Run the frozen wrist method on the independent I01-A smooth-shape fixtures."""
import argparse
import json
from pathlib import Path
import shutil
from time import sleep

from . import experiment
from .evaluate import score
from .history import GENERATED


OBJECTS=('oblate','ovoid','superellipsoid')
ARMS=('history-dp','rigid-dp','history-mcc')
START='start-03'


def prepare(root,parent):
    root.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((parent/'protocol.json').read_bytes())
    for name in ('hand-history.npz','history-sources.json'):
        if not (root/name).exists():
            shutil.copy2(parent/name,root/name)
    protocol.update(cases=[[o,START] for o in OBJECTS],arms=ARMS,sources={},
        stage='external smooth shapes, side start C; identical frozen planner',
        source_collection='i01_a_smooth_20261121',additional_pear=None,branch_experiment=None)
    for o in OBJECTS:
        case,surface=experiment.scenario(o,START)
        protocol['sources'][o+'/'+START]={k:experiment.digest(p) for k,p in {
            'model':case/'model.mjb','initial':case/'initial.npz','offline_surface':surface}.items()}
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_bytes())!=json.loads(json.dumps(protocol)):
        raise ValueError('Transfer protocol changed')
    experiment.save_json(root/'protocol.json',protocol)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','score'))
    p.add_argument('--root',type=Path,default=experiment.ROOT)
    p.add_argument('--arm',choices=ARMS)
    args=p.parse_args()
    # I01-A fair and smooth use the same fixture layout and named native body.
    # This separate process selects only the fixture archive; run() verifies its hashes.
    experiment.FAIR=GENERATED/'i01_a_smooth_20261121'
    root=args.root/'smooth-transfer'
    if args.action=='prepare':
        prepare(root,args.root)
        return
    if args.action=='run':
        if args.arm is None:
            p.error('--arm is required for run')
        for obj in OBJECTS:
            experiment.run(root,obj,START,args.arm,experiment.BUDGET)
    else:
        jobs=[(o,a) for o in OBJECTS for a in ARMS]
        while jobs:
            for o,a in jobs.copy():
                if (root/'runs'/o/START/a/'run.json').is_file():
                    score(root,o,START,a)
                    jobs.remove((o,a))
            if jobs:
                sleep(4)


if __name__=='__main__':
    main()
