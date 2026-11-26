"""Score the five 240 s finger-execution pairs against the published PARK AUDE runs."""

import argparse
import csv
import json
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PARK.identity import digest, read

from Module.palmreach6.__main__ import FIXTURES, ROOT

ROOT_OUT = ROOT / "outputs/active-3-paired"
EVALUATOR = ROOT / "outputs/PARK/models/evaluator"
OBJECTS = {"pear": "梨", "mustard": "芥末瓶", "cracker": "饼干盒", "bleach": "漂白剂瓶", "coffee": "咖啡罐"}
CONDITIONS = {"planned": "AUDE 规划手指", "single_all": "single_all 主动调指", "compliance": "仅底层柔顺"}
SECONDS, HZ = 240, 100
TICKS = SECONDS * HZ
ENDPOINT = "signed-240s.ply"
METRICS = (
    "c25", "f5", "ge2", "ge3", "c10", "precision5", "recall5", "surface_error_mm", "chamfer_mm2",
    "moving", "ge2_given_moving", "ge3_given_moving", "stable2", "stable3", "moving_stable2",
    "moving_stable3", "overload", "longest2_s", "longest3_s", "wall_s", "plan_s", "gp_s", "ik_s",
    "inference_s", "gpis_score_s",
)
CONTROL = dict(seconds=SECONDS, period=90, servo_hz=500, contact_log_hz=HZ, planner_feedback="physical",
               planner_variables=22, planning_object_geometries=0, wrist_mcc=False, finger_mcc=False)


def official_scores():
    with (ROOT / "PARK/scores.csv").open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["method"] == "AUDE" and int(r["seconds"]) == SECONDS]
    if len(rows) != len(OBJECTS) or {r["object"] for r in rows} != set(OBJECTS):
        raise ValueError("PARK/scores.csv must contain exactly one 240 s AUDE row per object")
    return {r["object"]: r for r in rows}


def match_official(score, official):
    if score["seed"] != int(official["seed"]):
        raise ValueError(f"Wrong AUDE seed for {score['object']}")
    for field in ("c10", "c25", "f5", "precision5", "recall5", "surface_error_mm", "chamfer_mm2"):
        is_ratio = field not in ("surface_error_mm", "chamfer_mm2")
        expected = float(official[field + "_pct" if is_ratio else field])
        np.testing.assert_allclose(score[field] * (100 if is_ratio else 1), expected, rtol=1e-13, atol=1e-12,
                                   err_msg=f"{score['object']} / {field} vs PARK/scores.csv")


def validate_controller(run, obj, condition, seed, checkpoint):
    sig = run["signature"]
    expected = dict(CONTROL, object=obj, seed=seed, initial=run["initial_sha256"])
    if (run["steps"] != TICKS or run["seed"] != seed or run["planning_object_geometries"] != 0
            or any(sig.get(key) != value for key, value in expected.items())):
        raise ValueError(f"Unmatched AUDE controller or time budget for {obj}/{condition}")
    if condition == "single_all":
        policy = run["policy"]
        step = .015 if obj == "mustard" else .020
        if (sig.get("finger_policy") not in ("single_all", "single_all_seed20261116")
                or policy["checkpoint_sha256"] != checkpoint or sig["checkpoint_sha256"] != checkpoint
                or policy["inference_seed"] != seed or sig.get("max_finger_step_rad", .020) != step
                or sig.get("lost_target_retry_radius_mm", 0) != 0):
            raise ValueError(f"Unmatched frozen finger policy for {obj}")
    elif condition == "compliance":
        if ("policy" in run or sig.get("execute_planned_fingers") is not False
                or sig.get("cartesian_tracking") is not False or sig["finger_reference_source"] is not None
                or not sig["intrinsic_nullspace_rest"] or sig["policy_queries"] != 0):
            raise ValueError(f"Expected the target-free AUDE torque controller for {obj}")
    elif condition != "planned" or "policy" in run or sig.get("hand") != "LEAP":
        raise ValueError(f"Expected native AUDE planned fingers at {obj}/{condition}")


def qualified(trace):
    forces, normals = trace["pad_force_n"], trace["normals_outward"]
    raw = trace["pad_valid"]
    if raw.shape != (TICKS + 1, 4) or raw.dtype != bool:
        raise ValueError("Expected the initial state followed by 24,000 four-finger measurements")
    np.testing.assert_array_equal(raw, (forces >= .2) & (np.linalg.norm(normals, axis=2) > .99))
    load = trace["region_load_n"].max(axis=1)
    return raw & (load < 8.)[:, None], load


def stable_bouts(valid, n):
    stable = np.zeros(len(valid), dtype=bool)
    longest = 0
    for fingers in combinations(range(4), n):
        mask = valid[:, fingers].all(axis=1)
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        for first, stop in edges.reshape(-1, 2):
            length = stop - first
            longest = max(longest, length)
            if length >= HZ:
                stable[first:stop] = True
    return stable, float(longest / HZ)


def contacts_and_motion(qpos, eligible, load):
    valid = eligible[1:]
    count = valid.sum(axis=1)
    moving = HZ * np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1) >= .005
    stable2, longest2 = stable_bouts(valid, 2)
    stable3, longest3 = stable_bouts(valid, 3)
    return dict(ge2=float((count >= 2).mean()), ge3=float((count >= 3).mean()),
                moving=float(moving.mean()),
                ge2_given_moving=float((count[moving] >= 2).mean()) if moving.any() else None,
                ge3_given_moving=float((count[moving] >= 3).mean()) if moving.any() else None,
                stable2=float(stable2.mean()), stable3=float(stable3.mean()),
                moving_stable2=float((moving & stable2).mean()), moving_stable3=float((moving & stable3).mean()),
                overload=float((load[1:] >= 8.).mean()), longest2_s=longest2, longest3_s=longest3)


def acquisition_times(run, plans, timing):
    if [row["tick"] for row in plans] != list(range(0, TICKS, CONTROL["period"])):
        raise ValueError("Missing AUDE planning cycle")
    result = dict(wall_s=run["wall_s"], inference_s=0.)
    for field, source in (("plan_s", "plan_ms"), ("gp_s", "gp_direction_ms"), ("ik_s", "ik_ms")):
        values = [row[source] for row in plans]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError(f"Invalid {source} in planning log")
        result[field] = sum(values) / 1000
    if "policy" in run:
        with np.load(timing, allow_pickle=False) as data:
            values = data["inference_ms"]
        if (len(values) != run["policy"]["queries"] or not len(values)
                or not np.isfinite(values).all() or min(values) < 0):
            raise ValueError("Unmatched finger-policy timings")
        result["inference_s"] = float(values.sum() / 1000)
    return result


def validate_inputs(folder, obj, condition, official_folder, origin, fixture):
    spec = fixture["objects"][obj]
    seed = spec["seed"]
    run = read(folder / "run.json")
    validate_controller(run, obj, condition, seed, fixture["checkpoint"])
    for field, filename in (("model_sha256", "model.mjb"), ("offline_mesh_sha256", "surface.obj")):
        if run[field] != spec[filename] or digest(FIXTURES / obj / filename) != spec[filename]:
            raise ValueError(f"Unmatched {filename} at {folder}")
    start = FIXTURES / obj / "initial.npz"
    with np.load(start, allow_pickle=False) as saved:
        initial = sha256(saved["integration_state"].tobytes()).hexdigest()
    if (spec["start"] != "start-02" or digest(start) != spec["initial.npz"]
            or not (run["initial_sha256"] == spec["integration_state"] == origin["initial_sha256"] == initial)):
        raise ValueError(f"Unmatched starting state at {folder}")
    baseline_sig, sig = origin["signature"], run["signature"]
    if (sig["source"] != baseline_sig["source"] or
            any(sig["code"].get(k) != v for k, v in baseline_sig["code"].items())):
        raise ValueError(f"AUDE planner differs from the PARK acquisition for {obj}")
    required = {"trace.npz", "plans.jsonl"} | ({"timing.npz"} if condition == "single_all" else set())
    if required != set(run["hashes"]) or run["trace_sha256"] != run["hashes"]["trace.npz"]:
        raise ValueError(f"Unmatched raw file inventory at {folder}")
    for name, fingerprint in run["hashes"].items():
        if digest(folder / name) != fingerprint:
            raise ValueError(f"Acquisition checksum mismatch: {folder / name}")
    with np.load(folder / "trace.npz", allow_pickle=False) as trace, np.load(
            official_folder / "trace.npz", allow_pickle=False) as original:
        for key in ("qpos", "pad_valid", "pad_force_n", "region_load_n", "points_m", "normals_outward"):
            # A rerun may have a different wall clock. Every physical row of planned must still match.
            same_run = condition == "planned" and run["trace_sha256"] != origin["trace_sha256"]
            np.testing.assert_array_equal(trace[key] if same_run else trace[key][0],
                                          original[key] if same_run else original[key][0], err_msg=key)
        np.testing.assert_allclose(trace["pad_force_n"][0], spec["pad_force_n"], rtol=0, atol=1e-8)
        if condition == "compliance" and (trace["goal_joints_rad"].shape != (TICKS + 1, 0)
                                         or trace["cartesian_tracking_error_m"].any()):
            raise ValueError(f"External finger target in the compliance group at {folder}")
        if condition == "single_all" and trace["single_all_raw_rad"].shape != (TICKS + 1, 16):
            raise ValueError(f"Missing finger commands at {folder}")
    return run


def score_trial(folder, obj, condition, *, recompute=False):
    """Reproduce PARK's endpoint score and attach timing and qualified-contact evidence."""
    import torch
    from PARK.core.common import write
    from PARK.core.study import archived
    from PARK.evaluation import metrics

    from Module.wrist_active.evaluate import Reference, reconstruct

    torch.set_num_threads(1)
    folder = Path(folder)
    fixture_file = FIXTURES / "fixtures.json"
    release = read(ROOT / "PARK/release.json")
    if digest(fixture_file) != release["runtime_dependencies"][fixture_file.relative_to(ROOT).as_posix()]:
        raise ValueError("The fixtures differ from the published PARK start-02 fixtures")
    fixture = read(fixture_file)
    seed = fixture["objects"][obj]["seed"]
    official_dir, official_raw, baseline, origin = archived(obj, SECONDS, seed, "AUDE")
    run = validate_inputs(folder, obj, condition, official_raw, origin, fixture)
    identity = dict(object=obj, method=condition, seed=seed,
                    trace_sha256=run["trace_sha256"], plans_sha256=run["hashes"]["plans.jsonl"],
                    initial_sha256=run["initial_sha256"], model_sha256=run["model_sha256"],
                    reference_sha256=run["offline_mesh_sha256"], evaluator_sha256=baseline["signature"]["metrics"],
                    scoring_gpis_sha256=baseline["signature"]["gpis"], report_code_sha256=digest(Path(__file__)),
                    official_aude_trace_sha256=origin["trace_sha256"], run_sha256=digest(folder / "run.json"))
    receipt = folder / "score.json"
    if receipt.exists() and not recompute:
        saved = read(receipt)
        if (all(saved.get(k) == value for k, value in identity.items())
                and all((folder / f).is_file() and digest(folder / f) == saved.get(key)
                        for f, key in ((ENDPOINT, "endpoint_sha256"), ("coverage.npz", "coverage_sha256")))):
            if condition == "planned":
                match_official(saved, official_scores()[obj])
            return saved
    with np.load(folder / "trace.npz", allow_pickle=False) as trace:
        valid, load = qualified(trace)
        t, fingers = np.nonzero(valid)
        xyz = trace["points_m"][t, fingers]
        normals = trace["normals_outward"][t, fingers]
        q = trace["qpos"]
        np.testing.assert_allclose(run["wall_s"], trace["wall_s"][-1], rtol=0, atol=1e-9)
    if not np.isfinite(q).all() or not np.isfinite(xyz).all():
        raise ValueError(f"Nonfinite position in qualified input at {folder}")
    extras = contacts_and_motion(q, valid, load)
    plans = [json.loads(line) for line in (folder / "plans.jsonl").read_text(encoding="utf-8").splitlines()]
    extras.update(acquisition_times(run, plans, folder / "timing.npz"))

    with TemporaryDirectory(dir=folder) as temp:
        stage = Path(temp)
        native_dir = official_dir if condition == "planned" else stage
        if condition == "planned":
            native = baseline
        else:
            meta = dict(method=condition, object=obj, seconds=SECONDS, seed=seed, source=".",
                        trace_sha256=run["trace_sha256"], initial_sha256=run["initial_sha256"],
                        model_sha256=run["model_sha256"])
            native = metrics.score(folder, native_dir, meta, EVALUATOR)
        reference = Reference(FIXTURES / obj / "surface.obj")
        curves, registered = reference.area(xyz, normals, t, TICKS)
        reference.neighbors.cache_clear()
        with (native_dir / "coverage-motion.csv").open(encoding="utf-8", newline="") as stream:
            sampled = list(csv.DictReader(stream))
        np.testing.assert_array_equal([float(r["seconds"]) for r in sampled], np.arange(SECONDS + 1))
        for curve, name in zip(curves, ("c10", "c25"), strict=True):
            np.testing.assert_allclose(curve[::HZ], [float(r[name]) for r in sampled], rtol=0, atol=1e-13)
        if native["registered"] != registered:
            raise ValueError(f"Contact registration mismatch at {folder}")
        # PARK records F@5 on the float64 mesh before PLY encoding. Recover its original
        # precision/recall counts and the measured endpoint scoring time with that exact call.
        rebuilt = stage / "endpoint.ply"
        result = reconstruct(xyz, normals, reference, rebuilt, "signed", signed_device="cuda")
        if result["status"] != "ok" or digest(rebuilt) != digest(native_dir / ENDPOINT):
            raise ValueError(f"Endpoint did not reproduce the canonical scored mesh at {folder}")
        final = native["final"]
        np.testing.assert_allclose([curves[0, -1], curves[1, -1], result["fscore"], result["mean_error_mm"]],
                                   [final[k] for k in ("c10", "c25", "f5", "surface_error_mm")],
                                   rtol=0, atol=1e-12)
        np.savez_compressed(stage / "coverage.npz", time_s=np.arange(TICKS + 1) / HZ,
                            c10=curves[0], c25=curves[1])
        scores = dict(**identity, step=(.015 if obj == "mustard" else .020) if condition == "single_all" else None,
                      **{k: final[k] for k in ("c25", "c10", "f5", "surface_error_mm", "chamfer_mm2")},
                      precision5=result["precision"], recall5=result["recall"], gpis_score_s=result["compute_s"],
                      **extras, endpoint_sha256=digest(rebuilt), coverage_sha256=digest(stage / "coverage.npz"),
                      archive_folder=f"{obj}/{condition}", registered=registered)
        if condition == "planned":
            match_official(scores, official_scores()[obj])
        rebuilt.replace(folder / ENDPOINT)
        (stage / "coverage.npz").replace(folder / "coverage.npz")
        write(receipt, scores)
    return scores


def summarize(rows, *, require_original=False):
    indexed = {(r["object"], r["method"]): r for r in rows}
    if len(rows) != 15 or set(indexed) != {(o, c) for o in OBJECTS for c in CONDITIONS}:
        raise ValueError("The five-object comparison requires all 15 completed runs")
    official = official_scores()
    origins = {}
    for obj in OBJECTS:
        baseline = indexed[obj, "planned"]
        match_official(baseline, official[obj])
        original = baseline["trace_sha256"] == baseline["official_aude_trace_sha256"]
        origins[obj] = "published_recording" if original else "verified_physical_rerun"
        if require_original and not original:
            raise ValueError(f"The published comparison must reuse the official AUDE recording for {obj}")
        for condition in CONDITIONS:
            current = indexed[obj, condition]
            if any(current[key] != baseline[key] for key in ("seed", "initial_sha256", "model_sha256",
                   "reference_sha256", "scoring_gpis_sha256", "evaluator_sha256", "official_aude_trace_sha256")):
                raise ValueError(f"Unmatched comparison for {obj}/{condition}")
    if len({r["trace_sha256"] for r in rows}) != len(rows):
        raise ValueError("Repeated trajectory in the 15-run comparison")
    means = {}
    for condition in CONDITIONS:
        means[condition] = {}
        for name in METRICS:
            values = [indexed[obj, condition][name] for obj in OBJECTS]
            means[condition][name] = float(np.mean(values)) if None not in values else None
    return dict(experiment="Active-3", seconds=SECONDS, control_hz=500, contact_hz=HZ,
                planning_period_s=.9, primary_seeds_per_object=1, start="start-02",
                protocol=dict(contact_force_min_n=.2, normal_norm_gt=.99, region_peak_load_lt_n=8.,
                              contact_time_denominator_ticks=TICKS, contact_time_excludes_initial_state=True,
                              coverage_geodesic_radius_mm=[10, 25], registration_max_mm=5,
                              registration_min_normal_cosine=.5, coverage_weight="reference vertex area",
                              reconstruction="signed GPIS on qualified touches up to and including the endpoint",
                              fscore_threshold_mm=5, samples_per_surface=50000, reconstruction_sampling_seed=408,
                              reference_sampling_seed=409, aggregate="unrounded equal-weight object mean"),
                official_aude_scores="PARK/scores.csv", planned_source=origins, conditions=CONDITIONS, objects={
                    obj: dict(label=label, seed=indexed[obj, "planned"]["seed"],
                              metrics={c: {k: indexed[obj, c][k] for k in METRICS} for c in CONDITIONS})
                    for obj, label in OBJECTS.items()}, macro_mean=means)


def pear_videos(root, rows):
    records = {}
    for row in rows:
        if row["object"] != "pear":
            continue
        condition = row["method"]
        receipt = read(root / "media-evidence" / "pear" / condition / "video.json")
        movie = f"pear-{condition}-4x.mp4"
        poster = f"pear-{condition}.webp"
        site_video = f"media/active-2/{movie}"
        site_poster = f"images/active-2/{poster}"
        expected = dict(object="pear", condition=condition, seed=row["seed"], trace_sha256=row["trace_sha256"],
                        initial_sha256=row["initial_sha256"], model_sha256=row["model_sha256"],
                        surface_sha256=row["reference_sha256"], coverage_sha256=row["coverage_sha256"],
                        endpoint_sha256=row["endpoint_sha256"], endpoint_c25=row["c25"], endpoint_f5=row["f5"],
                        source_hz=HZ, motion_tick_range=[0, TICKS], speed=4, fps=25, frame_count=1525,
                        movie_sha256=digest(root / "site-overlay" / site_video),
                        poster_sha256=digest(root / "site-overlay" / site_poster))
        snapshots = receipt["checkpoints"]
        if (any(receipt[k] != v for k, v in expected.items()) or
                [s["s"] for s in snapshots] != list(range(0, SECONDS + 1, 15)) or
                snapshots[-1]["mesh_sha256"] != row["endpoint_sha256"]):
            raise ValueError(f"Pear replay failed the score, timeline or media checksum check for {condition}")
        records[condition] = dict(video=f"media/{movie}", poster=f"media/{poster}", site_video=site_video,
                                  site_poster=site_poster, frames=receipt["frame_count"], fps=receipt["fps"],
                                  speed=receipt["speed"], simulation_s=SECONDS, sha256=receipt["movie_sha256"],
                                  poster_sha256=receipt["poster_sha256"], trace_sha256=row["trace_sha256"],
                                  endpoint_sha256=row["endpoint_sha256"])
    return records


def main():
    from PARK.core.common import table, write

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_OUT, help="Directory with runs/{object}/{condition}")
    parser.add_argument("--object", choices=OBJECTS)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--recompute", action="store_true", help="Ignore verified per-run scoring receipts")
    parser.add_argument("--with-media", action="store_true",
                        help="Include all three verified pear replays from media-evidence and site-overlay")
    args = parser.parse_args()
    if args.with_media and (args.object or args.condition):
        parser.error("--with-media applies to the complete comparison")
    rows = []
    for obj in (args.object,) if args.object else OBJECTS:
        for condition in (args.condition,) if args.condition else CONDITIONS:
            score = score_trial(args.root / "runs" / obj / condition, obj, condition, recompute=args.recompute)
            rows.append(score)
            print(f"{obj}/{condition} C25 {score['c25']:.4%}, F@5 {score['f5']:.4%}", flush=True)
    if not args.object and not args.condition:
        result = summarize(rows, require_original=args.with_media)
        out = args.root / "report"
        index = ("object", "method", "seed", "step", *METRICS)
        csv_rows = [{**{k: r[k] for k in index}, **r} for r in rows]
        if args.with_media:
            result["pear_video"] = pear_videos(args.root, rows)
        table(out / "trials.csv", csv_rows)
        write(out / "results.json", result)


if __name__ == "__main__":
    main()
