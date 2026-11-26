"""Verify the Active-3 trajectory archive or its extracted runs and the published tables."""

import argparse
import csv
import hashlib
import io
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import Path as ZipPath
from zipfile import ZipFile

import numpy as np
from PARK.identity import read

from .report import (
    CONDITIONS,
    ENDPOINT,
    FIXTURES,
    HZ,
    OBJECTS,
    ROOT,
    TICKS,
    acquisition_times,
    contacts_and_motion,
    qualified,
    summarize,
    validate_controller,
)


def trial(archive, obj, condition):
    prefix = f"{obj}/{condition}/"
    run = json.loads(archive.read(prefix + "run.json"))
    scores = json.loads(archive.read(prefix + "portable-scores.json"))
    raw = archive.read(prefix + "trace.npz")
    if hashlib.sha256(raw).hexdigest() != run["trace_sha256"]:
        raise ValueError(f"Trace checksum mismatch in {prefix}")
    with np.load(io.BytesIO(raw), allow_pickle=False) as trace:
        # Row zero holds the common initial state. Time fractions use 24,000 subsequent samples.
        valid = trace["pad_valid"][1:] & (trace["region_load_n"][1:].max(axis=1) < 8)[:, None]
        if valid.shape != (24000, 4):
            raise ValueError(f"Unexpected duration or finger count in {prefix}")
        count = valid.sum(axis=1)
    with np.load(io.BytesIO(archive.read(prefix + "coverage.npz")), allow_pickle=False) as coverage:
        c25 = float(coverage["c25"][-1])

    precision, recall = scores["precision5"], scores["recall5"]
    f5 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    values = (c25, f5, (count >= 2).mean(), (count >= 3).mean())
    expected = (scores["coverage25"], scores["f5"], scores["light2_s"] / 240, scores["light3_s"] / 240)
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-10, err_msg=prefix)
    fixture = (run["seed"], run["initial_sha256"], run["model_sha256"], run["offline_mesh_sha256"])
    return fixture, values


def legacy_archive(archive):
    """Keep the old archive readable under its own fixtures and scores."""
    groups = {mode: [] for mode in ("planned", "single_all", "compliance")}
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("object", "method", "c25", "f5", "ge2", "ge3"))
    for obj in OBJECTS:
        paired = set()
        for mode, rows in groups.items():
            fixture, values = trial(archive, obj, mode)
            paired.add(fixture)
            rows.append(values)
            writer.writerow((obj, mode, *values))
        if len(paired) != 1:
            raise ValueError(f"Nonmatching fixtures or inference seeds for {obj}")
    for mode, rows in groups.items():
        writer.writerow(("mean", mode, *np.mean(rows, axis=0)))


def checked_bytes(path, fingerprint):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != fingerprint:
        raise ValueError(f"Checksum mismatch for {path}")
    return raw


def verify_geometry(obj, trace, valid, mesh_bytes, score):
    import trimesh
    from scipy.sparse.csgraph import dijkstra

    from Module.wrist_active.evaluate import Reference, mesh_accuracy, reconstruct

    reference = Reference(FIXTURES / obj / "surface.obj")
    graph, areas = reference.graph, reference.ledger.vertex_area
    points, normals = trace["points_m"][valid], trace["normals_outward"][valid]
    registered = []
    for offset in range(0, len(points), 10000):
        sl = slice(offset, offset + 10000)
        vertex, distance = graph.oriented_nearest_vertices(points[sl], normals[sl])
        accepted = (distance <= .005) & (np.einsum("ij,ij->i", graph.normals[vertex], normals[sl]) >= .5)
        registered.extend(vertex[accepted])
    if len(registered) != score["registered"]:
        raise ValueError(f"Mismatch in the number of qualified registrations for {obj}")
    # A single multi-source solve recovers the endpoint without the evaluator's chronological curve.
    nearest = dijkstra(graph.adjacency, directed=False, indices=np.unique(registered),
                       min_only=True, limit=.025) if registered else np.full(len(areas), np.inf)
    for key, radius in (("c10", .010), ("c25", .025)):
        np.testing.assert_allclose(areas[nearest <= radius].sum() / areas.sum(), score[key], rtol=0, atol=1e-12)
    mesh = trimesh.load_mesh(io.BytesIO(mesh_bytes), file_type="ply", process=False)
    quality = mesh_accuracy(mesh, reference.truth)
    np.testing.assert_allclose(quality["chamfer_mm2"], score["chamfer_mm2"], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(quality["symmetric_mean_mm"], score["surface_error_mm"], rtol=0, atol=.001)
    at5 = quality["at_mm"]["5"]
    if (at5["precision"], at5["recall"]) != (score["precision5"], score["recall5"]):
        # Float32 PLY vertices can move a sampled distance across 5 mm. Rebuild the original float64 mesh.
        with TemporaryDirectory() as tmp:
            mesh_path = Path(tmp) / ENDPOINT
            at5 = reconstruct(points, normals, reference, mesh_path, "signed", signed_device="cuda")
            if at5["status"] != "ok" or mesh_path.read_bytes() != mesh_bytes:
                raise ValueError(f"Failed to reproduce the pre-encoding mesh for {obj}")
    np.testing.assert_allclose([at5["precision"], at5["recall"], at5["fscore"]],
                               [score[k] for k in ("precision5", "recall5", "f5")], rtol=0, atol=1e-13)


def verify_trial(folder, obj, condition, release, checkpoint, *, geometry=False,
                 controller_check=validate_controller, trace_check=None):
    score = json.loads((folder / "score.json").read_bytes())
    run = json.loads(checked_bytes(folder / "run.json", score["run_sha256"]))
    controller_check(run, obj, condition, release["seeds"][obj], checkpoint)
    fixtures_key = f"Module/generated/palm_progress_full_20261125/paired-fixtures/{obj}"
    expected = dict(object=obj, method=condition, archive_folder=f"{obj}/{condition}", seed=run["seed"],
                    initial_sha256=run["initial_sha256"], trace_sha256=run["trace_sha256"],
                    plans_sha256=run["hashes"]["plans.jsonl"],
                    model_sha256=release["runtime_dependencies"][fixtures_key + "/model.mjb"],
                    reference_sha256=release["runtime_dependencies"][fixtures_key + "/surface.obj"],
                    evaluator_sha256=release["workspace_dependencies"]["PARK/evaluation/metrics.py"],
                    scoring_gpis_sha256=release["workspace_dependencies"]["outputs/PARK/models/evaluator/gpis.pt"])
    if (any(score[k] != v for k, v in expected.items()) or run["hashes"]["trace.npz"] != score["trace_sha256"]
            or run["model_sha256"] != score["model_sha256"]
            or run["offline_mesh_sha256"] != score["reference_sha256"]):
        raise ValueError(f"Wrong object, fixture, trajectory or evaluator for {obj}/{condition}")
    mesh_bytes = checked_bytes(folder / ENDPOINT, score["endpoint_sha256"])
    plans = [json.loads(line) for line in checked_bytes(folder / "plans.jsonl", score["plans_sha256"]).splitlines()]
    with np.load(io.BytesIO(checked_bytes(folder / "trace.npz", score["trace_sha256"])), allow_pickle=False) as tr:
        valid, load = qualified(tr)
        if trace_check is not None:
            evidence = trace_check(run, tr, plans)
            if any(score.get(k) != v for k, v in evidence.items()):
                raise ValueError(f"Condition-specific evidence differs for {obj}/{condition}")
        if geometry:
            verify_geometry(obj, tr, valid, mesh_bytes, score)
        qpos = tr["qpos"]
        values = contacts_and_motion(qpos, valid, load)
        np.testing.assert_allclose(tr["wall_s"][-1], run["wall_s"], rtol=0, atol=1e-9)
        initial = tuple(tr[k][0].tobytes() for k in ("qpos", "pad_force_n", "region_load_n", "points_m",
                                                      "normals_outward", "pad_valid"))
    timing = (io.BytesIO(checked_bytes(folder / "timing.npz", run["hashes"]["timing.npz"]))
              if "policy" in run else None)
    values.update(acquisition_times(run, plans, timing))
    with np.load(io.BytesIO(checked_bytes(folder / "coverage.npz", score["coverage_sha256"])),
                 allow_pickle=False) as coverage:
        np.testing.assert_array_equal(coverage["time_s"], np.arange(TICKS + 1) / HZ)
        for key in ("c10", "c25"):
            curve = coverage[key]
            if (curve.shape != (TICKS + 1,) or not np.isfinite(curve).all() or
                    (np.diff(curve) < 0).any() or curve[0] < 0 or curve[-1] > 1):
                raise ValueError(f"Invalid {key} curve at {folder}")
            values[key] = curve[-1]
    p, r = score["precision5"], score["recall5"]
    if not 0 <= p <= 1 or not 0 <= r <= 1 or not score["gpis_score_s"] > 0:
        raise ValueError(f"Invalid endpoint precision, recall or timing at {folder}")
    values["f5"] = 2 * p * r / (p + r) if p + r else 0.
    for key, actual in values.items():
        if actual is None and score[key] is None:
            continue
        np.testing.assert_allclose(actual, score[key], rtol=0, atol=1e-12,
                                   err_msg=f"{obj}/{condition}/{key}")
    sig = run["signature"]
    pairing = (initial, sig["source"], *(sig["code"][k] for k in ("aude.py", "author_control.py", "route_free.py")))
    return score, pairing


def verify_report(path, rows, recomputed):
    published = read(path / "results.json")
    for field, expected in recomputed.items():
        if published.get(field) != expected:
            raise ValueError(f"Published results.json differs in {field}")
    if "pear_video" in published:
        videos = published["pear_video"]
        if set(videos) != set(CONDITIONS):
            raise ValueError("Incomplete pear video register")
        for row in (r for r in rows if r["object"] == "pear"):
            movie = videos[row["method"]]
            if (movie["trace_sha256"] != row["trace_sha256"] or movie["endpoint_sha256"] != row["endpoint_sha256"]
                    or (movie["frames"], movie["fps"], movie["speed"], movie["simulation_s"]) != (1525, 25, 4, 240)):
                raise ValueError(f"Video points to a different acquisition at pear/{row['method']}")
    with (path / "trials.csv").open(encoding="utf-8-sig", newline="") as stream:
        table = list(csv.DictReader(stream))
    indexed = {(r["object"], r["method"]): r for r in rows}
    if len(table) != len(indexed) or {(r["object"], r["method"]) for r in table} != set(indexed):
        raise ValueError("Unexpected object or condition in trials.csv")
    for row in table:
        source = indexed[row["object"], row["method"]]
        expected = {k: str(v) if v is not None else "" for k, v in source.items()}
        if row != expected:
            raise ValueError(f"Published trials.csv differs for {row['object']}/{row['method']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="trajectories.zip or the extracted runs directory")
    parser.add_argument("--report", type=Path, help="Also verify published results.json and trials.csv")
    parser.add_argument("--allow-reproduced-planned", action="store_true",
                        help="Accept local planned reruns whose physics were verified during full scoring")
    parser.add_argument("--geometry", action="store_true",
                        help="Independently score raw contacts and saved meshes using the restored PARK fixtures")
    args = parser.parse_args()
    release = read(ROOT / "PARK/release.json")
    checkpoint = read(ROOT / "PARK/contract.json")["finger_policy"]["checkpoint_sha256"]
    rows = []
    with (ZipFile(args.archive) if args.archive.is_file() else nullcontext(None)) as archive:
        root = ZipPath(archive) if archive else args.archive
        if (root / "runs").is_dir():
            root = root / "runs"
        if (archive and not (root / "pear" / "planned" / "score.json").is_file()
                and "pear/planned/portable-scores.json" in archive.namelist()):
            if args.report or args.geometry:
                parser.error("A historical trajectory ZIP cannot verify the current Active-3 tables")
            print("Historical Active-2 archive with its original fixtures", file=sys.stderr)
            legacy_archive(archive)
            return
        for obj in OBJECTS:
            pairings = set()
            for condition in CONDITIONS:
                row, pairing = verify_trial(root / obj / condition, obj, condition, release, checkpoint,
                                            geometry=args.geometry)
                rows.append(row)
                pairings.add(pairing)
            if len(pairings) != 1:
                raise ValueError(f"Initial measurements or AUDE planner differ for {obj}")
    result = summarize(rows, require_original=not args.allow_reproduced_planned)
    if args.report:
        verify_report(args.report, rows, result)
    metrics = ("c25", "f5", "ge2", "ge3", "precision5", "recall5")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("object", "method", *metrics))
    for r in rows:
        writer.writerow((r["object"], r["method"], *(r[k] for k in metrics)))
    for condition, mean in result["macro_mean"].items():
        writer.writerow(("mean", condition, *(mean[k] for k in metrics)))


if __name__ == "__main__":
    main()
