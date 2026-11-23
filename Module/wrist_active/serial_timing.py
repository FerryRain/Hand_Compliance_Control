"""Serial end-to-end wall time including the same signed GPIS mesh writer."""
import argparse
from pathlib import Path
import shutil
from time import perf_counter, sleep

import torch

from .evaluate import read_trace
from .experiment import ROOT, run, save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    root=args.root/'serial-60s'
    root.mkdir(exist_ok=True)
    for name in ('hand-history.npz','history-sources.json','protocol.json'):
        if not (root/name).exists():
            shutil.copy2(args.root/name,root/name)
    order=[('ellipsoid',a) for a in ('aude-mcc','history-mcc','history-dp')]+[
           ('meat-can',a) for a in ('history-dp','history-mcc','aude-mcc')]
    save_json(root/'timing-protocol.json',dict(seconds=60.,start='start-01',order=order,concurrency=1,
        measure='setup + closed-loop acquisition + trace writing + signed final mesh writing; offline scoring excluded'))
    if args.prepare:
        return
    full=args.root/'pear-full'
    while (full/'long-protocol.json').exists() and not (full/'long-results.json').exists():
        sleep(4)
    torch.set_num_threads(1)
    for obj,arm in order:
        folder=root/'runs'/obj/'start-01'/arm
        record=folder/'elapsed.json'
        if record.exists():
            continue
        attempt=root
        # A timed acquisition must run inside this stopwatch, including on retry.
        retry=0
        while (attempt/'runs'/obj/'start-01'/arm/'run.json').exists():
            retry+=1
            attempt=root/f'retry-{retry}'
            attempt.mkdir(exist_ok=True)
            for name in ('hand-history.npz','protocol.json'):
                if not (attempt/name).exists():
                    shutil.copy2(root/name,attempt/name)
        stamp=perf_counter()
        receipt=run(attempt,obj,'start-01',arm,60.)
        through_trace=perf_counter()-stamp
        trace=read_trace(attempt/'runs'/obj/'start-01'/arm/'trace.npz',6000)
        valid=trace['pad_valid']&(trace['region_load_n'].max(1)<8.)[:,None]
        # Include downsampling and mesh export. Offline scoring does not run in this timer.
        from ..acitive_02.full_signed import signed_gpis
        from ..i04_geometric_traversal import fingertip_r_gpis as ridge
        p,n=trace['points_m'][valid],trace['normals_outward'][valid]
        wall=perf_counter()
        ids=ridge.voxel_representatives(p)
        if len(ids)>ridge.CONTACT_BUDGET:
            ids=ids[ridge.farthest_indices(p[ids],ridge.CONTACT_BUDGET)]
        mesh,_=signed_gpis(p[ids],n[ids])
        mesh.export(folder/'signed-final.ply')
        mesh_s=perf_counter()-wall
        save_json(record,dict(obj=obj,arm=arm,sim_s=60.,acquisition_wall_s=receipt['wall_to_last_observation_s'],
            setup_s=receipt['setup_s'],planning_s=receipt['planning_s'],mesh_wall_s=mesh_s,
            wall_through_trace_s=through_trace,total_wall_s=perf_counter()-stamp))
    print('All serial timing cases complete.',flush=True)


if __name__=='__main__':
    main()
