"""Load compact IK pose libraries after obsolete native trajectories are removed.

Only constructor seed inputs are adapted. Archived controllers, source
snapshots, models, measured coverage and GP contact observations are untouched.
"""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
from scipy.spatial import cKDTree


def read_library(case, kind):
    case = Path(case)
    meta = json.loads((case/'retained_ik_seeds.json').read_text())
    entry = meta['libraries'][kind]
    source = case/entry['file']
    assert source.resolve().is_relative_to(case.resolve())
    assert sha256(source.read_bytes()).hexdigest() == entry['sha256']
    if entry.get('source_metadata'):
        assert sha256((case/entry['source_metadata']).read_bytes()).hexdigest() == entry['source_metadata_sha256']
    with np.load(source, allow_pickle=False) as z:
        q, points, fingers = z['seed_q'].copy(), z['seed_points'].copy(), z['seed_finger'].copy()
    assert q.shape == (len(points), 23) and points.shape == (len(fingers), 3)
    assert np.isfinite(q).all() and np.isfinite(points).all()
    assert np.all((fingers >= 0) & (fingers < 4))
    return q, points, fingers


def assign_library(planner, library):
    planner.seed_q, points, planner.seed_finger = library
    planner.seed_tree = cKDTree(points)


def install():
    """Use the byte-checked pose cache only when the original logs are absent."""
    from .whole_hand_surface_planner import WholeHandSurfacePlanner
    from .whole_hand_completion import CompletionSurfacePlanner
    if getattr(WholeHandSurfacePlanner, '_retained_seed_adapter', False):
        return
    original_init = WholeHandSurfacePlanner.__init__
    original_load = CompletionSurfacePlanner.load_cooperative_seeds

    def initialize(self, handles, graph, seed_case=None):
        fallback = (seed_case is not None and
            not (Path(seed_case)/'independent_native_audit/verified_preview.npz').exists() and
            (Path(seed_case)/'retained_ik_seeds.json').exists())
        original_init(self, handles, graph, None if fallback else seed_case)
        if fallback:
            assign_library(self, read_library(seed_case, 'surface'))

    def load(self, case):
        case = Path(case)
        summary = json.loads((case/'summary.json').read_text())
        if (all((case/c['file']).exists() for c in summary['trace_chunks'])
                or not (case/'retained_ik_seeds.json').exists()):
            return original_load(self, case)
        assign_library(self, read_library(case, 'completion'))

    WholeHandSurfacePlanner.__init__ = initialize
    CompletionSurfacePlanner.load_cooperative_seeds = load
    WholeHandSurfacePlanner._retained_seed_adapter = True


def main():
    import argparse
    import runpy
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    install()
    sys.argv = [args.module, *args.arguments]
    runpy.run_module(args.module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
