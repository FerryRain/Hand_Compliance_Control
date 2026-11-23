"""Freeze and execute the wrist-only comparison from the retained initial states."""
import argparse
import json
from pathlib import Path

from .experiment import ACTIVE2, CASES, ROOT, digest, save_json, seed_for
from .sweep_experiment import OUT, SMOOTH, hashes, run
from .sweep_score import ref_mesh, score


def jobs():
    # Same physics seed/state on pear. The second DP noise seed was reserved
    # before launching this matrix; no new planner settings on transfers.
    spec=[('primary','runs','pear','active-02','sweep-dp',1129.32,None,False)]
    for arm in ('sweep-mcc','route-dp','no-recontact-dp'):
        spec.append(('ablations','runs','pear','active-02',arm,240.,None,False))
    spec.append(('replicate','new-seed','pear','active-02','sweep-dp',600.,20261122,False))
    for obj,start in CASES:
        spec.append(('main-'+start,'runs',obj,start,'sweep-dp',240.,None,False))
    for obj in SMOOTH:
        spec.append(('smooth','runs',obj,'start-03','sweep-dp',240.,None,False))
    spec.append(('paced','paced','pear','active-02','sweep-dp',120.,None,True))
    keys=('group','series','obj','start','arm','seconds','seed','paced')
    return [dict(zip(keys,row)) for row in spec]


def freeze(root):
    root.mkdir(parents=True,exist_ok=True)
    refs = {}
    for r in jobs():
        obj,start = r['obj'],r['start']
        mesh = ref_mesh(obj,start)
        case = ACTIVE2 if obj=='pear' else mesh.parent
        refs[obj+'/'+start] = {name:digest(case/name) for name in ('model.mjb','initial.npz')}
        refs[obj+'/'+start]['evaluation_mesh'] = digest(mesh)
    baseline={arm:digest(ACTIVE2/arm/'trace.npz') for arm in ('aude-mcc','aude-single-all-mcc')}
    spec=dict(jobs=jobs(), wrist_code=hashes(),fixtures=refs,baseline=baseline,
              scoring_sha256=digest(Path(__file__).with_name('sweep_score.py')),
              selector='Two development iterations on pear and meat-can start-01. Frozen before this matrix.',
              robot_input='known fixture origin, robot FK, current four fingertip samples and five region loads',
              planning='five closed-form wrist candidates, zero optimized finger variables; 5 Hz, 50 ms lead',
              policy='released single_all, 10 Hz requests, 50 ms lead and buffered 100 Hz consumption',
              coverage='geodesic 10 mm, >=0.2 N measured head, every region <8 N',
              reconstruction='same oriented signed GPIS/F-score at 5 mm and fixed 50k mesh sample seeds',
              hardware_test='paced software loop includes MuJoCo; no physical robot',
              previous_protocol=digest(ROOT/'protocol.json'))
    file = root/'protocol.json'
    if file.exists() and json.loads(file.read_bytes())!=spec:
        raise ValueError('Frozen suite differs from the working code or data; choose a new output.')
    if not file.exists():
        save_json(file,spec)
    return spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('freeze','run','score','references'))
    p.add_argument('--root',type=Path,default=OUT)
    p.add_argument('--group')
    a=p.parse_args()
    spec=freeze(a.root)
    if a.action=='freeze':
        return
    if a.action=='references':
        for arm in ('history-dp','aude-mcc','aude-single-all-mcc'):
            source=ROOT/'pear-full/runs/pear/active-02'/arm if arm=='history-dp' else ACTIVE2/arm
            score(source,a.root/'references/pear/active-02'/arm,'pear','active-02',[60.,120.,240.,600.,1129.32])
        return
    for r in spec['jobs']:
        if a.group and r['group']!=a.group:
            continue
        folder=a.root/r['series']/r['obj']/r['start']/r['arm']
        if a.action=='run':
            record=run(folder,r['obj'],r['start'],r['arm'],r['seconds'],r['paced'],r['seed'])
            expected=spec['fixtures'][r['obj']+'/'+r['start']]
            if (record['initial_file_sha256']!=expected['initial.npz'] or record['model_sha256']!=expected['model.mjb']
                    or record['seed']!=(r['seed'] or seed_for(r['obj'],r['start']))):
                raise ValueError('Run used a different initial state or policy seed')
        else:
            times=[s for s in (60.,120.,240.,600.,1129.32) if s<=r['seconds']]
            score(folder,folder,r['obj'],r['start'],times)


if __name__=='__main__':
    main()
