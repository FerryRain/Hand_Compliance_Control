"""Fresh whole-object traversal with finite-surface cooperative contact fits."""
import argparse
import json
from .whole_hand_surface_traversal import WholeHandTraversal
from .whole_hand_finite_surface import FiniteSurfacePlanner
from .whole_hand_geometry_cache import CommandGeometryCache
from .whole_hand_belly_probe import snapshot_sources


class FiniteWholeHandTraversal(WholeHandTraversal):
    def __init__(self,output,geometry,seed_case,seconds):
        super().__init__(output,geometry,seed_case,seconds)
        self.planner=FiniteSurfacePlanner(self.h,self.graph,seed_case)
        self.command_cache=CommandGeometryCache(self.planner.safety)
        self.planner.safety.command_collision_reason=self.command_cache.command_reason
        manifest=json.loads((self.out/'manifest.json').read_text())
        manifest.update(protocol='i04-whole-hand-surface.development.v4',
            finite_surface_support_required=True,contact_count_precedes_arm_distance=True,
            command_query_cache='Exact q bytes; unchanged original pairwise guard; independent original guard at audit',
            source_hashes=snapshot_sources(self.out))
        (self.out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--geometry',required=True)
    p.add_argument('--seed-case',required=True);p.add_argument('--seconds',type=float,default=4800.)
    a=p.parse_args();FiniteWholeHandTraversal(a.output,a.geometry,a.seed_case,a.seconds).run()
