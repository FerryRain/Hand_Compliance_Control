"""Prefer nearby cooperative wrist solutions before global reseeding."""
import argparse
import json
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
from .whole_hand_finite_surface import FiniteSurfacePlanner
from .whole_hand_finite_traversal import FiniteWholeHandTraversal
from .whole_hand_geometry_cache import CommandGeometryCache
from .whole_hand_belly_probe import snapshot_sources


class ContinuitySurfacePlanner(FiniteSurfacePlanner):
    def find_patch(self,root,p,n,*,clearance=.0002,budget_s=8.,preferred=None):
        points,normals=self.project_pads(root)
        order=np.argsort(np.linalg.norm(points-p,axis=1)).tolist()
        if preferred is not None:order=[int(preferred)]+[i for i in order if i!=preferred]
        attempts=[];local=None
        for finger in order:
            targets=points.copy();ns=normals.copy();targets[finger]=p;ns[finger]=n
            patch=self.fit_patch(root,targets,ns,anchor=finger,clearance=clearance,
                iterations=80,reproject=2,anchor_weight=1.8)
            count=int(patch['expected_contacts'].sum())
            attempts.append(dict(seed='current_wrist_continuation',anchor=finger,reason=patch['reason'],contacts=count))
            if patch['reason'] is None:
                if count>=3:patch['attempts']=attempts;return patch,attempts
                if local is None:local=patch
        patch,global_attempts=super().find_patch(root,p,n,clearance=clearance,budget_s=budget_s,preferred=preferred)
        attempts+=global_attempts
        if patch is None:patch=local
        if patch is not None:patch['attempts']=attempts
        return patch,attempts


class ContinuityWholeHandTraversal(FiniteWholeHandTraversal):
    def __init__(self,output,geometry,seed_case,seconds):
        super().__init__(output,geometry,seed_case,seconds)
        self.planner=ContinuitySurfacePlanner(self.h,self.graph,seed_case)
        self.command_cache=CommandGeometryCache(self.planner.safety)
        self.planner.safety.command_collision_reason=self.command_cache.command_reason
        manifest=json.loads((self.out/'manifest.json').read_text())
        manifest.update(protocol='i04-whole-hand-surface.development.v5',current_wrist_continuation_before_global_seeds=True,
            pivot_uses_finite_surface_targets=True,source_hashes=snapshot_sources(self.out))
        (self.out/'manifest.json').write_text(json.dumps(manifest,indent=2))

    def pivot_route(self,degrees):
        anchor=self.anchor;center=self.points[anchor];normal=self.normals[anchor]
        qs=[self.last.copy()];points=[self.points.copy()];normals=[self.normals.copy()];expected=[]
        for angle in np.linspace(0.,np.radians(degrees),6)[1:]:
            raw=center+Rotation.from_rotvec(angle*normal).apply(self.points-center)
            projected,_,_=trimesh.proximity.closest_point(self.graph.mesh,raw-self.h.object_position_m)
            pts=projected+self.h.object_position_m;ns=np.array([self.planner.smooth_normal(p) for p in projected])
            pts[anchor]=center;ns[anchor]=normal
            patch=self.planner.fit_patch(qs[-1],pts,ns,anchor=anchor,iterations=40,flexible=False)
            if patch['reason'] or not self.planner.safety.edge_free(qs[-1],patch['q']):return None
            qs.append(patch['q']);points.append(patch['points']);normals.append(patch['normals']);expected.append(patch['expected_contacts'])
        return dict(q=np.array(qs),points=np.array(points),normals=np.array(normals),anchor=anchor,
            distance=np.linspace(0.,abs(np.radians(degrees))*.004/.08,6),expected_contacts=np.array(expected),goal=-1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--geometry',required=True)
    p.add_argument('--seed-case',required=True);p.add_argument('--seconds',type=float,default=4800.)
    a=p.parse_args();ContinuityWholeHandTraversal(a.output,a.geometry,a.seed_case,a.seconds).run()
