"""Finite geometry-only six-region check for the cooperative whole traversal."""
import argparse
import json
from pathlib import Path
import numpy as np
from .full_surface import build_scene
from .whole_hand_surface_planner import WholeHandSurfacePlanner


def run(output, seed_case, extremes=False, finite=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    h,g,home,quality=build_scene(out)
    if finite:
        from .whole_hand_finite_surface import FiniteSurfacePlanner
        planner=FiniteSurfacePlanner(h,g,seed_case)
    else:planner=WholeHandSurfacePlanner(h,g,seed_case)
    records=[];qs=[];points=[];normals=[];vertices=[]
    labels=[]
    for axis,sign in (((1,1),(1,-1)) if extremes else ((2,1),(0,1),(2,-1),(0,-1),(1,1),(1,-1))):
        eligible=g.required_vertices[(g.normals[g.required_vertices,axis]*sign)>.75]
        center=g.vertices_m[eligible].mean(axis=0)
        chosen=(eligible[np.argmax(sign*g.vertices_m[eligible,axis])] if extremes else
                eligible[np.argmin(np.linalg.norm(g.vertices_m[eligible]-center,axis=1))])
        patch,attempts=planner.find_patch(home,g.vertices_m[chosen]+h.object_position_m,
            planner.smooth_normal(g.vertices_m[chosen]),budget_s=25. if extremes else 15.)
        labels.append([axis,sign]);record=dict(axis=axis,sign=sign,vertex=int(chosen),position_local_m=g.vertices_m[chosen].tolist(),
            extreme=extremes,accepted=patch is not None,attempts=attempts)
        if patch is not None:
            qs.append(patch['q']);points.append(patch['points']);normals.append(patch['normals']);vertices.append(chosen)
            record.update(anchor=patch['anchor'],expected_contacts=patch['expected_contacts'].tolist(),details=patch['details'])
        records.append(record);print('REGION',json.dumps(record),flush=True)
        (out/'survey.json').write_text(json.dumps(dict(geometry_only=True,coverage_granted=False,records=records),indent=2))
    np.savez_compressed(out/'patches.npz',q=qs,points=points,normals=normals,vertices=vertices)
    (out/'model.xml').write_text(h.xml,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed-case',required=True)
    p.add_argument('--extremes',action='store_true');p.add_argument('--finite',action='store_true')
    a=p.parse_args();run(a.output,a.seed_case,a.extremes,a.finite)
