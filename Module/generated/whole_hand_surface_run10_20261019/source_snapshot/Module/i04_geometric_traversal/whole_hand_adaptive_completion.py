"""Try nonpenetrating approach margins for residual cooperative contact sites.

The margin only changes geometric candidate generation. Actual MCC contact,
force limits, both command guards and measured coverage are unchanged.
"""
import argparse
import json
import numpy as np
from .whole_hand_completion import CompletionSurfacePlanner,CompletionWholeHandTraversal
from .whole_hand_geometry_cache import CommandGeometryCache
from .whole_hand_belly_probe import snapshot_sources


class AdaptiveCompletionPlanner(CompletionSurfacePlanner):
    def find_patch(self,root,p,n,*,clearance=.0002,budget_s=8.,preferred=None):
        patch,attempts=super().find_patch(root,p,n,clearance=clearance,budget_s=budget_s,preferred=preferred)
        if patch is not None or clearance!=.0002:return patch,attempts
        recorded=[dict(item,candidate_clearance_m=clearance) for item in attempts]
        best=None;best_score=None
        for margin in (.001,.002,.003):
            candidate,tries=super().find_patch(root,p,n,clearance=margin,budget_s=budget_s,preferred=preferred)
            recorded.extend(dict(item,candidate_clearance_m=margin) for item in tries)
            if candidate is None:continue
            assert candidate['reason'] is None
            count=int(candidate['expected_contacts'].sum())
            motion=float(np.linalg.norm(candidate['q'][:7]-root[:7])+.15*np.linalg.norm(candidate['q'][7:]-root[7:]))
            score=(-count,motion)
            if best_score is None or score<best_score:
                best=candidate;best_score=score;best['search_clearance_m']=margin
            if count==4:break
        if best is not None:best['attempts']=recorded
        return best,recorded


class AdaptiveCompletionTraversal(CompletionWholeHandTraversal):
    def __init__(self,output,geometry,seed_case,seconds,resume):
        super().__init__(output,geometry,seed_case,seconds,resume)
        self.planner=AdaptiveCompletionPlanner(self.h,self.graph,seed_case)
        self.planner.load_cooperative_seeds(resume)
        self.command_cache=CommandGeometryCache(self.planner.safety)
        self.planner.safety.command_collision_reason=self.command_cache.command_reason
        manifest=json.loads((self.out/'manifest.json').read_text())
        manifest.update(protocol='i04-whole-hand-surface.development.v8',
            fallback_candidate_clearances_m=[.001,.002,.003],
            fallback_scope='Only after the default finite cooperative candidate search fails; original geometry predicate still mandatory',
            measured_contact_and_coverage_unchanged=True,source_hashes=snapshot_sources(self.out))
        (self.out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--geometry',required=True)
    p.add_argument('--seed-case',required=True);p.add_argument('--seconds',type=float,default=7000.);p.add_argument('--resume',required=True)
    a=p.parse_args();AdaptiveCompletionTraversal(a.output,a.geometry,a.seed_case,a.seconds,a.resume).run()
