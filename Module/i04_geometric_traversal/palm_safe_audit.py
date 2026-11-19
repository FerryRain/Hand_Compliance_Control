"""Independent native replay and palm clearance audit for the repaired B.

Preserves the established independent contact/coverage and dual-guard replay.
Adds a separate palm predicate from compiled native contact distances, with
no import of the runtime PalmMonitor or its classification code.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
import json
from pathlib import Path

from . import whole_hand_parallel_audit as upstream
from .b_full_portable import register, initialize_worker


PROTOCOL = 'i04.b.palm-collision.mcc.v1'
EXPECTED_PARTS = ('fr3_leap_mount_adapter', 'palm_lower_collision',
                  'palm_1_fsr_geom', 'palm_2_fsr_geom', 'palm_3_fsr_geom', 'palm_4_fsr_geom')


def audit(case, *, workers=2, follow=False, output_name='independent_native_audit'):
    case = Path(case).resolve()
    manifest = json.loads((case/'manifest.json').read_text())
    assert manifest['protocol'] in (PROTOCOL, 'i04.b.palm-collision-recovery.mcc.v2')
    assert tuple(manifest['palm_collision_parts']) == EXPECTED_PARTS
    assert manifest['palm_stop_clearance_m'] == .001
    assert manifest['palm_contact_query_margin_m'] == .006
    register(case)
    source = inspect.getsource(upstream.audit)
    additions = {
        'high_steps=high_steps+1 if f.max()>12. else 0':
            'high_steps=high_steps+1 if f.max()>40. else 0',
        'assert not unsafe and arm<=.5 and high_steps<10':
            'assert not unsafe and arm<=.5 and high_steps<100',
        "finished=(case/'summary.json').exists()":
            "finished=(case/'run_complete.json').exists() or (not follow and (case/'summary.json').exists())",
        "endpoint=case/'final_state.npz' if (case/'final_state.npz').exists() else case/'checkpoint.npz'":
            "endpoint=case/'checkpoint.npz'",
        'c=_native_context(case);m,d=c.model,c.data':
            '''c=_native_context(case);m,d=c.model,c.data
    palm_ids=set(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in EXPECTED_PARTS)
    assert -1 not in palm_ids
    object_body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'fr3_e05_object')
    object_ids=set(map(int,np.flatnonzero((m.geom_bodyid==object_body)&(m.geom_contype!=0))))
    assert len(object_ids)==256
    assert all(m.geom_contype[g]==8 and m.geom_conaffinity[g]==2 for g in palm_ids)
    assert all(m.body_contype[int(m.geom_bodyid[g])]&8 for g in palm_ids)
    minimum_palm=.006;palm_samples=0''',
        'live=np.r_[safe,safe.any()];gaps=np.where(live,0.,gaps+.002);longest=np.maximum(longest,gaps)':
            '''palm_distance=.006
            for contact in d.contact:
                ga,gb=int(contact.geom1),int(contact.geom2)
                if (ga in palm_ids and gb in object_ids) or (gb in palm_ids and ga in object_ids):
                    palm_distance=min(palm_distance,float(contact.dist))
            check('palm_clearance',palm_distance,tr['palm_clearance_m'][k])
            assert palm_distance>=.001,('PALM_CLEARANCE',row,palm_distance)
            minimum_palm=min(minimum_palm,palm_distance);palm_samples+=1
            live=np.r_[safe,safe.any()];gaps=np.where(live,0.,gaps+.002);longest=np.maximum(longest,gaps)''',
        'source_snapshot_count=len(manifest[\'source_hashes\']),maximum_errors=maximum,':
            '''source_snapshot_count=len(manifest['source_hashes']),maximum_errors=maximum,
        independent_palm_samples=palm_samples,minimum_palm_clearance_m=minimum_palm,
        palm_contact_geometry_verified=True,palm_parts=EXPECTED_PARTS,''',
    }
    for before, after in additions.items():
        if source.count(before) != 1:
            raise RuntimeError(f'Auditor adapter source no longer matches: {before}')
        source = source.replace(before, after)
    namespace = dict(vars(upstream), initialize_guard_worker=initialize_worker,
                     EXPECTED_PARTS=EXPECTED_PARTS)
    exec(compile(source, __file__, 'exec'), namespace)
    result = namespace['audit'](case, workers=workers, follow=follow, output_name=output_name)
    out = case/output_name
    (out/'effective_palm_auditor.py').write_text(source, encoding='utf-8')
    (out/'palm_audit_adapter.py').write_bytes(Path(__file__).read_bytes())
    result.update(protocol=manifest['protocol'],
        effective_palm_auditor_sha256=sha256(source.encode('utf-8')).hexdigest(),
        palm_audit_adapter_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.case, workers=args.workers, follow=args.follow), indent=2), flush=True)
