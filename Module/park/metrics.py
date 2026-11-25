"""Score physical measurements on the native reference and signed reconstruction protocol."""
import csv
import json
from pathlib import Path

import numpy as np
import trimesh

from Module.i04_geometric_traversal.fingertip_surface_report import mesh_accuracy
from Module.palmreach6.__main__ import FIXTURES
from Module.wrist_active.evaluate import Reference, reconstruct

from .gpis import GPIS
from .settings import DT, FINGERS, digest, rotation, write_json


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    if not len(values):
        return None
    return dict(n=len(values), mean=float(np.mean(values)), p50=float(np.median(values)),
                p95=float(np.percentile(values, 95)), total=float(np.sum(values)))


def timing(source, method, initial):
    from .settings import Settings
    from .trajectory import Trajectory
    if method.startswith("park-"):
        cfg = Settings(**json.loads((source / "run.json").read_bytes())["signature"]["code"]["settings"])
        path = Trajectory(0, initial)
    columns = {"planning_ms": [], "gpis_query_ms": [], "gpis_update_ms": [], "candidate_ms": [],
               "kde_ms": [], "evaluation_ms": [], "global_ms": [], "gp_direction_ms": []}
    reasons = {}
    example = None
    with (source / "plans.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            for key, column in columns.items():
                field = "plan_ms" if method == "aude" and key == "planning_ms" else key
                val = row.get(field)
                if isinstance(val, (float, int)) and val > 0:
                    column.append(val)
            if "reason" in row:
                reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
            if row.get("reason") == "new_target" and (example is None or example["tick"] == 0):
                splice, tick = row["splice_tick"], row["apply_tick"]
                tail = path.revise(splice, np.asarray(row["target"]), cfg)
                prefix = path.at(np.arange(tick, splice))
                path = Trajectory(tick, np.vstack((prefix, tail.samples)))
                example = dict(row, trajectory=tail.samples[::max(1, len(tail.samples) // 80), :3].tolist())
    stats = {name: distribution(data) for name, data in columns.items()}
    if (source / "control-timing.npz").exists():
        with np.load(source / "control-timing.npz") as d:
            stats["servo_service_ms"] = distribution(d["service_ms"])
    return stats, reasons, example


def score(source, out, meta, models):
    source, out = Path(source), Path(out)
    model_id = digest(models / "gpis.pt")
    from . import plots
    signature = dict(trace=meta["trace_sha256"], gpis=model_id, metrics=digest(Path(__file__)),
                     plots=digest(Path(plots.__file__)), plans=digest(source / "plans.jsonl"))
    receipt = out / "metrics.json"
    if receipt.exists():
        saved = json.loads(receipt.read_bytes())
        if saved["signature"] == signature and all((out / f).exists() for f in saved["files"]):
            return saved
    if digest(source / "trace.npz") != meta["trace_sha256"]:
        raise ValueError("Acquisition trace failed checksum verification")
    obj, seconds = meta["object"], meta["seconds"]
    spec = json.loads((FIXTURES / "fixtures.json").read_bytes())["objects"][obj]
    ref = Reference(FIXTURES / obj / "surface.obj")
    with np.load(source / "trace.npz", allow_pickle=False) as f:
        tr = {k: f[k] for k in ("qpos", "points_m", "normals_outward", "pad_valid", "region_load_n", "pad_force_n")}
        if meta["method"].startswith("park-"):
            np.testing.assert_array_equal(f["published_preview"][1:-20], f["palm_command"][21:])
    q, raw, force = tr["qpos"], tr["pad_valid"], tr["pad_force_n"]
    np.testing.assert_allclose(force[0], spec["pad_force_n"], rtol=0, atol=1e-8)
    measured = (force >= .2) & (np.linalg.norm(tr["normals_outward"], axis=2) > .99)
    np.testing.assert_array_equal(raw, measured)
    peak = tr["region_load_n"].max(1)
    valid = raw & (peak < 8.)[:, None]
    t, f = np.nonzero(valid)
    xyz, normals = tr["points_m"][t, f], tr["normals_outward"][t, f]
    _, first = np.unique(np.floor(xyz / .0045).astype(int), axis=0, return_index=True)
    first.sort()
    first_ticks = t[first]
    n = len(q) - 1
    if n != seconds * 100 or not np.isfinite(q).all():
        raise ValueError("Incomplete physical trace")
    curves, registered = ref.area(xyz, normals, t, n)
    ref.neighbors.cache_clear()
    dp = np.diff(q[:, :3], axis=0)
    dr = (rotation(q[1:]) * rotation(q[:-1]).inv()).as_rotvec()
    lengths = np.r_[0., np.cumsum(np.linalg.norm(dp, axis=1))]
    turns = np.r_[0., np.cumsum(np.linalg.norm(dr, axis=1))]
    new = np.cumsum(valid.sum(1)) - valid[0].sum()
    ticks = np.unique(np.r_[np.arange(0, n + 1, 100), n])
    series = [dict(seconds=int(k) * DT, path_m=lengths[k], rotation_rad=turns[k],
                   c10=curves[0, k], c25=curves[1, k], valid_samples=int(new[k]),
                   new_voxels=int(np.searchsorted(first_ticks, k, "right") - (first_ticks == 0).sum()))
              for k in ticks]
    write_csv(out / "coverage-motion.csv", series)
    stats, reasons, example = timing(source, meta["method"], q[0, :7])
    fixed_queries = ref.truth[np.linspace(0, len(ref.truth) - 1, 384, dtype=int)]
    gp = GPIS.load(models / "gpis.pt")
    quality, last_mesh = [], None
    # Common causal checkpoints and an independent acquisition at each official budget.
    checkpoints = sorted({s for s in (1, 10, 30, 60, 120, 240, 360, 480, seconds) if s <= seconds})
    for s in checkpoints:
        k = s * 100
        count = np.searchsorted(t, k, "right")
        destination = out / f"signed-{s}s.ply"
        info = reconstruct(xyz[:count], normals[:count], ref, destination, "signed", signed_device="cuda")
        accuracy = {}
        if info["status"] == "ok":
            last_mesh = trimesh.load_mesh(destination, process=False)
            accuracy = mesh_accuracy(last_mesh, ref.truth)
        ix = first[first_ticks <= k]
        gp.condition(xyz[ix], normals[ix])
        variance = gp.query(fixed_queries)[1]
        quality.append(dict(seconds=s, path_m=float(lengths[k]), c25=float(curves[1, k]),
                            f5=info["fscore"], chamfer_mm2=accuracy.get("chamfer_mm2"),
                            surface_error_mm=info["mean_error_mm"], status=info["status"],
                            mean_reference_variance_m2=float(variance.mean())))
    write_csv(out / "reconstruction.csv", quality)
    end = quality[-1]
    loss = (raw[:-1] & ~raw[1:]).sum(0)
    raw_force = force[1:][raw[1:]]
    force_stats = dict(mean_n=float(raw_force.mean()), p95_n=float(np.percentile(raw_force, 95)),
                       max_n=float(raw_force.max())) if len(raw_force) else None
    digits = {name: dict(contact_ratio=float(raw[1:, i].mean()), qualified_ratio=float(valid[1:, i].mean()),
                         losses=int(loss[i]), losses_per_min=float(loss[i] * 60 / seconds))
              for i, name in enumerate(FINGERS)}
    final = dict(f5=end["f5"], chamfer_mm2=end["chamfer_mm2"], surface_error_mm=end["surface_error_mm"],
                 c25=float(curves[1, -1]), c10=float(curves[0, -1]), path_m=float(lengths[-1]),
                 rotation_rad=float(turns[-1]), valid_samples=int(new[-1]), new_voxels=series[-1]["new_voxels"],
                 contact_ratio=float(raw[1:].mean()), qualified_ratio=float(valid[1:].mean()),
                 losses_per_finger_min=float(loss.mean() * 60 / seconds), peak_load_n=float(peak.max()),
                 force_mean_n=force_stats["mean_n"] if force_stats else None,
                 force_p95_n=force_stats["p95_n"] if force_stats else None,
                 plan_p50_ms=stats["planning_ms"]["p50"] if stats["planning_ms"] else None)
    plots.trial_figure(out, meta, q, xyz[first], last_mesh, quality, series, example, gp)
    files = ["coverage-motion.csv", "reconstruction.csv", "exploration.png"]
    files += [f"signed-{r['seconds']}s.ply" for r in quality if r["status"] == "ok"]
    if example:
        files.append("wrist-selection.png")
    saved = dict(signature=signature, files=files, meta=meta, final=final, contact=digits, force=force_stats,
                 translation_axis_variation_m=np.abs(dp).sum(0).tolist(),
                 rotation_world_axis_variation_rad=np.abs(dr).sum(0).tolist(),
                 registered=registered, timing_ms=stats, decisions=reasons, checkpoints=quality)
    write_json(receipt, saved)
    return saved

