"""Recompute C10, C25 and mesh F@5 for a published formal trial."""

import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import trimesh

from Module.i04_geometric_traversal.fingertip_surface_report import mesh_accuracy
from Module.wrist_active.evaluate import Reference
from .__main__ import FIXTURES, OBJECTS


ARMS = {"palmreach-6": "profiled", "facing-4": "facing4", "aude": "aude"}


def verify(archive, obj, method, seconds):
    ref = Reference(FIXTURES / obj / "surface.obj")
    base = "Module/generated/retained_cracker_full_20261125"
    trial = f"full/{obj}/{ARMS[method]}/{seconds}s"
    scores = f"{base}/scores/{trial}"
    with ZipFile(archive) as bundle:
        with np.load(BytesIO(bundle.read(f"{base}/runs/{trial}/trace.npz")), allow_pickle=False) as trace:
            valid = trace["pad_valid"] & (trace["region_load_n"].max(axis=1) < 8)[:, None]
            if len(valid) != seconds * 100 + 1:
                raise ValueError("Trace duration differs from the requested budget")
            t, f = np.nonzero(valid)
            points = trace["points_m"][t, f]
            normals = trace["normals_outward"][t, f]
        coverage, _ = ref.area(points, normals, t, seconds * 100)
        with np.load(BytesIO(bundle.read(f"{scores}/curves.npz")), allow_pickle=False) as curves:
            np.testing.assert_allclose(coverage[:, ::100], [curves["c10"], curves["c25"]], rtol=0, atol=1e-8)
        mesh = trimesh.load_mesh(BytesIO(bundle.read(f"{scores}/signed-{seconds}s.ply")), file_type="ply", process=False)
        checkpoint, = json.loads(bundle.read(f"{scores}/score.json"))["checkpoints"]
    c10, c25 = coverage[:, -1] * 100
    f5 = 100 * mesh_accuracy(mesh, ref.truth)["at_mm"]["5"]["fscore"]
    np.testing.assert_allclose([c10, c25], [checkpoint["c10_pct"], checkpoint["c25_pct"]], rtol=0, atol=1e-6)
    np.testing.assert_allclose(f5, checkpoint["f5_pct"], rtol=0, atol=.1)
    ref.neighbors.cache_clear()
    return {"object": obj, "method": method, "seconds": seconds, "c10_pct": c10, "c25_pct": c25, "f5_pct": f5}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--object", choices=OBJECTS, required=True)
    parser.add_argument("--method", choices=ARMS, default="palmreach-6")
    parser.add_argument("--seconds", choices=(240, 600), type=int, required=True)
    opts = parser.parse_args()
    print(json.dumps(verify(opts.archive, opts.object, opts.method, opts.seconds)))
