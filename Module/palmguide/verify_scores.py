"""Recompute the published coverage, scored-mesh F@5 and global-decision gains."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import trimesh

from Module.i04_geometric_traversal.fingertip_surface_report import mesh_accuracy
from Module.wrist_active.evaluate import Reference
from Module.wrist_active.information_score import LENGTHS_M, NOISE_VARIANCE, gp_prefix, measured_cells
from Module.wrist_active.ycb_fixtures import OBJECTS, fixture


def verify(data, obj, method, seconds):
    ours = method == "palmguide"
    arm = "ours" if ours else "aude"
    folder = "trajectories" if ours else "comparison/active-1-b"
    endpoint = seconds * 100
    expected = json.loads((data / "metrics/reconstruction.json").read_bytes())
    curve = expected["objects"][obj][arm]
    reference = Reference(fixture(obj)[1])
    with ZipFile(data / folder / f"{seconds}s" / f"{obj}.zip") as archive:
        with np.load(BytesIO(archive.read("trace.npz")), allow_pickle=False) as trace:
            valid = trace["pad_valid"] & (trace["region_load_n"].max(axis=1) < 8)[:, None]
            if len(valid) != endpoint + 1:
                raise ValueError("Trace duration differs from the requested endpoint")
            tick, finger = np.nonzero(valid)
            points = trace["points_m"][tick, finger]
            normals = trace["normals_outward"][tick, finger]
            coverage = 100 * reference.area(points, normals, tick, endpoint)[0][1]
            np.testing.assert_allclose(coverage[::100], curve["coverage25"][:seconds+1], rtol=0, atol=1e-6)
            if seconds == 240:
                with archive.open("plans.jsonl") as lines:
                    rows = [json.loads(line) for line in lines]
                verify_decisions(data, obj, arm, trace, rows, endpoint)

    with ZipFile(data / "reconstructions/scored-surfaces.zip") as archive:
        with archive.open(f"{method}/{obj}/{seconds}s.ply") as f:
            mesh = trimesh.load_mesh(BytesIO(f.read()), file_type="ply", process=False)
    scored = mesh_accuracy(mesh, reference.truth)
    f5 = 100 * scored["at_mm"]["5"]["fscore"]
    record = next(r for r in curve["reconstruction"] if r["time_s"] == seconds)
    # The original scoring pipeline independently sampled the exported PLY and
    # allowed 0.1 percentage points and 0.01 mm against the in-memory score.
    for key, saved in (("fscore", "f5"), ("precision", "precision5"), ("recall", "recall5")):
        np.testing.assert_allclose(100 * scored["at_mm"]["5"][key], record[saved], rtol=0, atol=.1)
    np.testing.assert_allclose(scored["symmetric_mean_mm"], record["mean_error_mm"], rtol=0, atol=.01)
    print(f"{obj}, {method}, {seconds} s: coverage {coverage[-1]:.2f}%, mesh F@5 {f5:.2f}%")
    reference.neighbors.cache_clear()


def verify_decisions(data, obj, arm, trace, rows, endpoint):
    points, ticks = measured_cells(trace)
    prefix, _ = gp_prefix(points, LENGTHS_M[0], NOISE_VARIANCE)
    counts = np.searchsorted(ticks, np.arange(endpoint+1), side="right")
    info = prefix[counts] - prefix[counts[0]]
    if arm == "ours":
        start = [r["apply_tick"] for r in rows if r["mode"] == "explore" and
                 (r["global_event"] or {}).get("direction_chosen")]
    else:
        start = [r["tick"] for r in rows if r["mode"] == "explore"]
    start = np.asarray(start, dtype=int)
    end = np.r_[start[1:], endpoint]
    actual = np.column_stack((start / 100, end / 100, info[end] - info[start]))
    expected = json.loads((data / "metrics/information.json").read_bytes())["objects"][obj][arm]
    np.testing.assert_allclose(info[::100], expected["cumulative_nat"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(actual, expected["decisions"], rtol=0, atol=1e-6)
    print(f"{len(start)} global decisions verified; {actual[:,2].mean():.2f} nat per decision.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="Downloaded palmguide dataset directory")
    p.add_argument("--object", choices=OBJECTS, required=True)
    p.add_argument("--method", choices=("palmguide", "active-1-b"), default="palmguide")
    p.add_argument("--seconds", type=int, choices=(240, 600), default=240)
    args = p.parse_args()
    verify(args.data, args.object, args.method, args.seconds)


if __name__ == "__main__":
    main()
