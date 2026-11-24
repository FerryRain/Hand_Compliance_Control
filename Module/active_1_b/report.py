"""Summarize the five retained 240 s physical-feedback traces and export their meshes."""
import argparse
import csv
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import trimesh

from ..acitive_02.active_a import GraspEnvelope
from ..acitive_02.experiment import digest, read_json
from ..acitive_02.route_free_inputs import OBJECTS, load_trial

NAMES = dict(pear="梨", mustard="芥末瓶", cracker="饼干盒", bleach="漂白剂瓶", coffee="咖啡罐")
METRICS = ("coverage25", "coverage10", "precision5", "recall5", "f5", "surface_error_mm",
           "mean_coverage25_over_time", "qualified_cells", "information_nat", "wall_s",
           "plan_s", "gp_direction_s", "ik_s", "final_gpis_and_score_s")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def object_report(key, folder, fixtures, output):
    receipt = read_json(folder / "run.json")
    sig = receipt["signature"]
    if (sig["object"], sig["seconds"], sig["planner_feedback"], sig["planning_object_geometries"],
        sig["wrist_mcc"], sig["finger_mcc"]) != (key, 240, "physical", 0, False, False):
        raise ValueError(f"Unexpected experiment signature for {key}")
    for name, checksum in receipt["hashes"].items():
        if digest(folder / name) != checksum:
            raise ValueError(f"{key}/{name} failed its acquisition checksum")
    case, models, case_path, offline_mesh = load_trial(fixtures, key)
    if (case["integration_state"] != receipt["initial_sha256"]
        or digest(case_path / "model.mjb") != receipt["model_sha256"]
        or digest(offline_mesh) != receipt["offline_mesh_sha256"]):
        raise ValueError(f"Fixture mismatch for {key}")
    scores = read_json(folder / "portable-scores.json")
    plans = [json.loads(line) for line in (folder / "plans.jsonl").read_text(encoding="utf-8").splitlines()]
    if [p["tick"] for p in plans] != list(range(0, 24000, sig["period"])):
        raise ValueError(f"Unexpected planning ticks for {key}")
    with np.load(folder / "coverage.npz") as cr, np.load(folder / "trace.npz") as tr:
        c25, c10, time = cr["c25"], cr["c10"], cr["time_s"]
        q = tr["qpos"]
        if len(q) != 24001 or not np.isfinite(q).all():
            raise ValueError(f"Incomplete physics trace for {key}")
        np.testing.assert_allclose(time, np.arange(len(q)) * .01)
        if min(np.diff(c25).min(), np.diff(c10).min()) < -1e-12:
            raise ValueError(f"Coverage must be cumulative for {key}")
        np.testing.assert_allclose([c10[-1], c25[-1]], [scores["coverage"], scores["coverage25"]])
        force = tr["pad_force_n"][1:]
        load = tr["region_load_n"][1:].max(axis=1)
        eligible = tr["pad_valid"][1:] & (load < 8)[:, None]
        grasp = GraspEnvelope(q[0, 7:], models[2].limits)
        displacement = np.rad2deg(np.abs(q[:, 7:] - grasp.bound(q[:, 7:])).max(axis=1))
        curve = [dict(object=key, time_s=int(time[i]), coverage10=float(c10[i]), coverage25=float(c25[i]))
                 for i in range(0, len(time), 100)]
        reach = {}
        for threshold in (.25, .50, .70):
            hits = np.flatnonzero(c25 >= threshold)
            reach[f"{threshold:.0%}"] = float(time[hits[0]]) if len(hits) else None
        contacts = dict(at_least_two_eligible_s=float((eligible.sum(1) >= 2).sum() / 100),
                        at_least_three_eligible_s=float((eligible.sum(1) >= 3).sum() / 100),
                        three_positive_pads_s=float(((force > 0).sum(1) >= 3).sum() / 100),
                        overload_s=float((load >= 8).sum() / 100), peak_region_force_n=float(load.max()),
                        measured_pose_beyond_2deg_s=float((displacement[1:] > 2).sum() / 100),
                        p95_pose_deviation_deg=float(np.percentile(displacement, 95)))
    precision, recall = scores["precision5"], scores["recall5"]
    np.testing.assert_allclose(2 * precision * recall / (precision + recall), scores["f5"])
    mesh_path = folder / "signed-240s.ply"
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Invalid final mesh for {key}")
    (output / "meshes").mkdir(exist_ok=True)
    copyfile(mesh_path, output / "meshes" / f"{key}.ply")
    metrics = dict(coverage25=float(c25[-1]), coverage10=float(c10[-1]), precision5=precision,
                   recall5=recall, f5=scores["f5"], surface_error_mm=scores["final_surface_error_mm"],
                   mean_coverage25_over_time=float(c25[:-1].mean()), qualified_cells=scores["cells"],
                   information_nat=scores["total_nat"], wall_s=receipt["wall_s"],
                   plan_s=sum(p["plan_ms"] for p in plans) / 1000,
                   gp_direction_s=sum(p["gp_direction_ms"] for p in plans) / 1000,
                   ik_s=sum(p["ik_ms"] for p in plans) / 1000,
                   final_gpis_and_score_s=scores["final_gpis_and_score_s"])
    provenance = dict(seed=case["seed"], trace_sha256=receipt["trace_sha256"],
                      plans_sha256=receipt["hashes"]["plans.jsonl"], mesh_sha256=digest(mesh_path),
                      initial_state_sha256=receipt["initial_sha256"], physics_model_sha256=receipt["model_sha256"],
                      offline_mesh_sha256=receipt["offline_mesh_sha256"], acquisition_code=sig["code"])
    item = dict(object=key, name=NAMES[key], **metrics, first_coverage25_s=reach, contact=contacts,
                plan_count=len(plans), provenance=provenance)
    return item, curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="Directory containing one run per object")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []
    for key in OBJECTS:
        entry, curve = object_report(key, args.runs / key, args.fixtures, args.out)
        rows.append(entry)
        curves.extend(curve)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
    result = dict(experiment="active-1-b", simulated_seconds_per_object=240, trials_per_object=1,
                  upstream_commit="d89a6941fec699768e64fb6acc07439ab822e59f", objects=rows, macro_mean=mean)
    (args.out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
                                          + "\n", encoding="utf-8")
    table = [{k: r[k] for k in ("object", *METRICS)} for r in rows]
    write_csv(args.out / "metrics.csv", table + [dict(object="macro_mean", **mean)])
    write_csv(args.out / "coverage-1hz.csv", curves)


if __name__ == "__main__":
    main()
