"""Export the atlas's measured information and contact metrics from scored PARK trials."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from Module.acitive_02.active_gpis import NOISE_VARIANCE, VOXEL_M
from Module.acitive_02.experiment import digest
from Module.wrist_active import information_score
from Module.wrist_active.sweep import DT, LOAD_LIMIT


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "i04_experiment_repo/Module/generated/retained_cracker_full_20261125"
OBJECTS = ("coffee", "cracker", "mustard", "bleach", "pear")
METHODS = ("PARK", "PalmReach-6", "AUDE")
BUDGETS = (240, 600)
HZ = round(1 / DT)
LENGTH_M = .025
ZERO_NAT = 1e-10
COHORTS = {"four": OBJECTS[:4], "five": OBJECTS}
SUMMARY_FIELDS = (
    "new_info_nat", "new_cells", "nat_per_s", "issued_decisions", "nat_per_issued_decision",
    "zero_gain_decision_pct", "info_before_first_issued_nat", "two_pad_pct", "three_pad_pct",
    "no_qualified_s", "overload_s", "qualified_pad_seconds",
    "high_level_updates", "planning_median_ms", "planning_total_s",
)


def read(path):
    return json.loads(path.read_bytes())


def canonical_inputs(obj, method, seconds):
    seed = 20261120 if obj == "pear" else 20261122
    key = f"seed-{seed}"
    if method == "PARK":
        score = ROOT / "outputs/PalmReach-3-v3/trials" / f"{seconds}s" / obj / key / "score"
    else:
        score = ROOT / "outputs/PARK/experiments" / f"{seconds}s" / obj / method.lower() / key
    metrics = read(score / "metrics.json")
    meta = metrics["meta"]
    source = (score / meta["source"].replace("\\", "/")).resolve()
    expected_method = "PalmReach-3" if method == "PARK" else method.lower()
    if (meta["object"], meta["seed"], meta["method"], meta["seconds"]) != (obj, seed, expected_method, seconds):
        raise ValueError(f"Unexpected score identity at {score}")
    trace_sha = digest(source / "trace.npz")
    if trace_sha != metrics["signature"]["trace"] or trace_sha != meta["trace_sha256"]:
        raise ValueError(f"Scored trace and input disagree for {obj} {method}")
    plan_name = "global-plans.jsonl" if method == "PARK" else "plans.jsonl"
    plan_sha = digest(source / plan_name)
    if method == "PARK":
        receipt = read(source / "adapter.json")
        if receipt["signature"]["revision"] != "v3" or receipt["trace_sha256"] != trace_sha:
            raise ValueError(f"Expected measured-start release for {obj}")
        expected_plan = receipt["telemetry"][plan_name]
    else:
        expected_plan = metrics["signature"]["plans"]
    if plan_sha != expected_plan:
        raise ValueError(f"Scored decision log and input disagree for {obj} {method}")
    return source / "trace.npz", source / plan_name, dict(
        object=obj, method=method, seed=seed, seconds=seconds, trace_sha256=trace_sha,
        decisions_sha256=plan_sha, initial_sha256=meta["initial_sha256"],
    )


def measured_information(trace, seconds):
    end = seconds * HZ
    pads = trace["pad_valid"]
    peak = trace["region_load_n"].max(axis=1)
    normals, force, points = (trace[k] for k in ("normals_outward", "pad_force_n", "points_m"))
    if len(pads) != end + 1 or any(not np.isfinite(a).all() for a in (peak, normals, force, points)):
        raise ValueError("Invalid or incomplete physical tactile trace")
    np.testing.assert_array_equal(pads, (force >= .2) & (np.linalg.norm(normals, axis=2) > .99))
    light = pads & (peak < LOAD_LIMIT)[:, None]
    ticks, fingers = np.nonzero(light)
    xyz = points[ticks, fingers]
    _, first = np.unique(np.floor(xyz / VOXEL_M).astype(np.int64), axis=0, return_index=True)
    first.sort()
    first_ticks = ticks[first]
    initial = np.count_nonzero(first_ticks == 0)
    if initial != 4:
        raise ValueError("Expected four distinct initial measured tactile cells")
    # The retained atlas scorer uses the first physical sample in each voxel, in temporal order.
    prefix, _ = information_score.gp_prefix(xyz[first], LENGTH_M, NOISE_VARIANCE)
    counts = np.searchsorted(first_ticks, np.arange(end + 1), side="right")
    info = prefix[counts] - prefix[initial]
    pad_counts = light[1:].sum(axis=1)
    contact = dict(
        new_cells=int(counts[-1] - initial),
        two_pad_pct=float(100 * np.mean(pad_counts >= 2)),
        three_pad_pct=float(100 * np.mean(pad_counts >= 3)),
        no_qualified_s=float(np.count_nonzero(pad_counts == 0) / HZ),
        overload_s=float(np.count_nonzero(peak[1:] >= LOAD_LIMIT) / HZ),
        qualified_pad_seconds=float(pad_counts.sum() / HZ),
    )
    return info, contact


def issued_events(path, method, stop):
    """Use each planner's recorded handoff; a held or failed target starts no new interval."""
    events, planning_ms = [], []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            tick = row["tick" if method == "AUDE" else "apply_tick"]
            if tick >= stop:
                continue
            if method == "PARK":
                planning_ms.append(row["planning_ms"])
                if row["reason"] != "new_target":
                    continue
                target = dict(wrist_xyz_wxyz=row["target"], surface_target_m=row["point"])
            elif method == "PalmReach-6":
                if row["global_event"] is not None:
                    planning_ms.append(row["global_ms"])
                event = row["global_event"] or {}
                if row["mode"] != "explore" or not event.get("direction_chosen"):
                    continue
                target = dict(palm_target_m=event["palm_target_m"], direction=event["direction"])
            else:
                planning_ms.append(row["plan_ms"])
                if row["mode"] != "explore":
                    continue
                target = dict(targets_m=row["targets_m"], directions=row["directions"])
            events.append((tick, target))
    ticks = np.array([t for t, _ in events], dtype=int)
    if np.any(ticks < 0) or np.any(np.diff(ticks) <= 0):
        raise ValueError(f"Unordered applied decision ticks in {path}")
    if not planning_ms or not np.isfinite(planning_ms).all() or min(planning_ms) < 0:
        raise ValueError(f"Missing or invalid measured high-level planning times in {path}")
    timing = dict(high_level_updates=len(planning_ms), planning_median_ms=float(np.median(planning_ms)),
                  planning_total_s=sum(planning_ms) / 1000)
    return ticks, [target for _, target in events], timing


def check_historical(trial, cumulative, scorer_sha):
    arm = "profiled" if trial["method"] == "PalmReach-6" else "aude"
    folder = ARCHIVE / "scores/full" / trial["object"] / arm / f"{trial['seconds']}s"
    archive = read(folder / "score.json")
    if (archive["seed"], archive["inputs"]["trace"], archive["inputs"]["plans"],
            archive["inputs"]["information_evaluator"]) != (
            trial["seed"], trial["trace_sha256"], trial["decisions_sha256"], scorer_sha):
        raise ValueError(f"Archived information inputs differ for {folder}")
    if digest(folder / "curves.npz") != archive["files"]["curves.npz"]:
        raise ValueError(f"Unverified baseline information curve at {folder}")
    with np.load(folder / "curves.npz", allow_pickle=False) as curve:
        np.testing.assert_allclose(cumulative, curve["new_info_nat"], rtol=0, atol=1e-7)
    checkpoint = next(r for r in archive["checkpoints"] if r["seconds"] == trial["seconds"])
    for field in SUMMARY_FIELDS:
        if field in checkpoint:
            np.testing.assert_allclose(trial[field], checkpoint[field], rtol=0, atol=1e-7, err_msg=field)


def score_case(obj, method, seconds, cache, scorer_sha, exporter_sha):
    trace_path, plan_path, identity = canonical_inputs(obj, method, seconds)
    key = dict(**identity, scorer_sha256=scorer_sha, exporter_sha256=exporter_sha)
    dest = cache / f"{obj}-{method}-{seconds}.json"
    if dest.is_file():
        saved = read(dest)
        if saved["key"] == key:
            return saved["result"]
    with np.load(trace_path, allow_pickle=False) as trace:
        info, contact = measured_information(trace, seconds)
    ticks, targets, timing = issued_events(plan_path, method, seconds * HZ)
    edges = np.r_[ticks, seconds * HZ]
    gains = np.diff(info[edges])
    initial_gain = float(info[edges[0]])
    total = float(info[-1])
    np.testing.assert_allclose(initial_gain + gains.sum(), total, rtol=0, atol=1e-7)
    trial = dict(
        **identity, **contact, **timing, new_info_nat=total, nat_per_s=total / seconds,
        issued_decisions=len(gains), nat_per_issued_decision=float(gains.mean()) if len(gains) else None,
        zero_gain_decision_pct=float(100 * (gains <= ZERO_NAT).mean()) if len(gains) else None,
        info_before_first_issued_nat=initial_gain,
    )
    cumulative = info[::HZ]
    if method != "PARK":
        check_historical(trial, cumulative, scorer_sha)
    series = dict(
        object=obj, method=method, seed=identity["seed"], seconds=seconds,
        cumulative_nat=np.round(cumulative, 8).tolist(),
        decisions=[[float(a / HZ), float(b / HZ), round(float(g), 9)] for a, b, g in zip(ticks, edges[1:], gains)],
    )
    result = {"trial": trial, "series": series, "targets": targets}
    cache.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".tmp")
    partial.write_text(json.dumps(dict(key=key, result=result), allow_nan=False), encoding="utf-8")
    partial.replace(dest)
    print(f"{obj} · {method} · {seconds}s · {total:.2f} nat · {len(gains)} decisions", flush=True)
    return result


def equal_object_means(trials):
    means = []
    for cohort, objects in COHORTS.items():
        for seconds in BUDGETS:
            for method in METHODS:
                rows = [t for t in trials if t["object"] in objects and (t["method"], t["seconds"]) == (method, seconds)]
                if len({r["object"] for r in rows}) != len(objects):
                    raise ValueError(f"Incomplete {cohort} cohort")
                means.append(dict(cohort=cohort, method=method, seconds=seconds, n=len(rows), **{
                    k: float(np.mean([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None
                    for k in SUMMARY_FIELDS
                }))
    return means


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def decision_rows(results):
    for result in results:
        trial = result["trial"]
        tags = {k: trial[k] for k in ("object", "method", "seed", "seconds")}
        for i, (interval, target) in enumerate(zip(result["series"]["decisions"], result["targets"], strict=True), 1):
            start, end, gain = interval
            yield dict(**tags, decision_index=i, start_s=start, end_s=end, gain_nat=gain,
                       issued_target_json=json.dumps(target, separators=(",", ":")))


def export(output, cache):
    scorer_sha = digest(Path(information_score.__file__))
    exporter_sha = digest(Path(__file__))
    results = [score_case(obj, method, seconds, cache, scorer_sha, exporter_sha)
               for seconds in BUDGETS for obj in OBJECTS for method in METHODS]
    trials = [r["trial"] for r in results]
    for obj in OBJECTS:
        identities = {(t["seed"], t["initial_sha256"]) for t in trials if t["object"] == obj}
        if len(identities) != 1:
            raise ValueError(f"Methods or budgets used unmatched initial states for {obj}")
    protocol = dict(
        hz=HZ, finger_force_min_n=.2, all_region_loads_strictly_below_n=LOAD_LIMIT,
        voxel_mm=1000 * VOXEL_M, kernel="Matérn 3/2", length_mm=1000 * LENGTH_M,
        signal_variance=1, noise_variance=NOISE_VARIANCE, conditioned_initial_cells=4,
        cumulative_unit="nat", cumulative_stride_s=1, gain_zero_threshold_nat=ZERO_NAT,
        decision_window="gain is information at next applied decision (or endpoint) minus information at this applied decision",
        cohort_mean="equal weight per object, including nat per issued decision",
        planning_measurements={"PARK": "planning_ms for each recorded global planner update",
                               "PalmReach-6": "global_ms for each recorded global_event",
                               "AUDE": "plan_ms for every scheduled policy call including coupled wrist and finger planning"},
        evaluator_sha256=scorer_sha, exporter_sha256=exporter_sha,
    )
    bundle = dict(methods=METHODS, protocol=protocol, trials=trials, means=equal_object_means(trials),
                  series=[r["series"] for r in results])
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temp:
        stage = Path(temp)
        (stage / "information.json").write_text(
            json.dumps(bundle, ensure_ascii=False, allow_nan=False, separators=(",", ":")), encoding="utf-8")
        write_csv(stage / "information-summary.csv", trials[0].keys(), trials)
        write_csv(stage / "decisions.csv",
                  ("object", "method", "seed", "seconds", "decision_index", "start_s", "end_s", "gain_nat",
                   "issued_target_json"), decision_rows(results))
        for file in stage.iterdir():
            file.replace(output / file.name)
    print(f"Verified {len(trials)} matched trials and {sum(t['issued_decisions'] for t in trials)} decision windows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / ".tmp/park-final-assets/data/park")
    parser.add_argument("--cache", type=Path, default=ROOT / ".tmp/park-final-information-cache")
    args = parser.parse_args()
    export(args.out, args.cache)
