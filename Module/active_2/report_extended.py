"""Score the two additional AUDE conditions and assemble the 25-run Active-3 comparison."""

import argparse
import csv
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PARK.identity import digest, read

from . import report as base
from .verify import verify_trial

ADDITIONAL = {"aude_mcc": "AUDE 腕规划 + single_all/MCC", "aude_height": "AUDE + 掌轴距离约束"}
CONDITIONS = {**base.CONDITIONS, **ADDITIONAL}
ROOT_OUT = base.ROOT_OUT / "extended"
PAIR_FIELDS = ("seed", "initial_sha256", "model_sha256", "reference_sha256", "evaluator_sha256",
               "scoring_gpis_sha256", "official_aude_trace_sha256")
INITIAL_FIELDS = ("qpos", "pad_valid", "pad_force_n", "region_load_n", "points_m", "normals_outward")


def validate_controller(run, obj, condition, seed, checkpoint):
    if condition not in ADDITIONAL:
        raise ValueError(f"Unknown additional condition {condition}")
    mcc = condition == "aude_mcc"
    sig = run["signature"]
    expected = dict(base.CONTROL, condition=condition, object=obj, seed=seed, initial=run["initial_sha256"],
                    finger_mcc=mcc, execute_planned_fingers=not mcc)
    if (any(sig.get(k) != v for k, v in expected.items()) or run["seed"] != seed or
            run["steps"] != base.TICKS or run["planning_object_geometries"] != 0):
        raise ValueError(f"Unmatched AUDE controller or budget at {obj}/{condition}")
    inputs = read(base.ROOT / "PARK/release.json")["runtime_dependencies"]
    fixtures = base.FIXTURES.relative_to(base.ROOT).as_posix()
    if (run["initial_file_sha256"] != inputs[f"{fixtures}/{obj}/initial.npz"] or
            run["robot_sha256"] != inputs[f"{fixtures}/hand.mjb"]):
        raise ValueError(f"Unmatched released initial state file or robot for {obj}/{condition}")
    required = {"trace.npz", "plans.jsonl"}
    if mcc:
        required.add("timing.npz")
        policy = run["policy"]
        contract = read(base.ROOT / "PARK/contract.json")
        if (any(policy.get(k) != v for k, v in contract["finger_policy"].items())
                or policy["inference_seed"] != seed or sig["checkpoint_sha256"] != checkpoint
                or sig["finger_policy"] != policy["policy"] or sig["history"] != "current rollout only"
                or policy["handoff_misses"] != 0 or policy["skipped_queries"] != 0):
            raise ValueError(f"Unmatched deployed single_all wrist handoff or finger policy for {obj}")
        for path in ("Module/wrist_active/buffered_fingers.py", "Module/i04_geometric_traversal/fingertip_traversal.py",
                     "Module/i04_geometric_traversal/fingertip_dp_worker.py",
                     "Module/i04_geometric_traversal/fingertip_local_policy.py"):
            if sig["code"].get(Path(path).name) != contract["original_files"][path]:
                raise ValueError(f"Different deployed finger controller at {obj}: {path}")
    elif "policy" in run:
        raise ValueError(f"Unexpected learned hand policy in the constrained pure AUDE run for {obj}")
    if not required.issubset(run["hashes"]) or run["trace_sha256"] != run["hashes"]["trace.npz"]:
        raise ValueError(f"Incomplete acquisition files at {obj}/{condition}")


def condition_evidence(run, trace, plans):
    condition = run["signature"]["condition"]
    ticks = [p["tick"] for p in plans]
    if ticks != list(range(0, base.TICKS, base.CONTROL["period"])):
        raise ValueError(f"Incomplete AUDE planning log for {condition}")
    plan_index = np.arange(base.TICKS) // base.CONTROL["period"]
    wrist = np.asarray([p["wrist"] for p in plans])
    np.testing.assert_allclose(trace["goal_wrist"][1:], wrist[plan_index], rtol=0, atol=1e-10)
    if condition == "aude_height":
        from .height_evidence import audit

        return audit(run, trace, plans, plan_index)
    nominal, bias, command = (trace[k] for k in ("finger_nominal_rad", "mcc_bias_rad", "finger_command_rad"))
    limits = np.asarray(run["joint_limits_rad"])
    target = np.clip(nominal[1:] + bias[1:], limits[:, 0] + .01, limits[:, 1] - .01)
    preview = trace["policy_wrist_preview"]
    if (nominal.shape != (base.TICKS + 1, 16) or bias.shape != nominal.shape or command.shape != nominal.shape
            or preview.shape != (base.TICKS + 1, 7) or not np.isfinite(preview).all()
            or not np.isfinite(command).all() or not np.isfinite(bias).all()
            or np.max(np.abs(bias)) > .0400000001 or np.max(np.abs(np.diff(nominal, axis=0))) > .0180000001
            or np.asarray([p["aude_joints_unexecuted_rad"] for p in plans]).shape != (len(plans), 16)):
        raise ValueError("Invalid native policy command, preview, or MCC joint offsets")
    np.testing.assert_allclose(command[1:], target, rtol=0, atol=1e-10)
    np.testing.assert_allclose(trace["goal_joints_rad"], nominal, rtol=0, atol=1e-10)
    return dict(planning_cycles=len(plans), executed_fingers="buffered single_all + fingertip MCC position servos",
                planning_fingers_executed=False, policy_queries=run["policy"]["queries"],
                mcc_peak_abs_bias_rad=float(np.max(np.abs(bias[1:]))),
                mcc_nonzero_bias_frames=int(np.any(bias[1:] != 0, axis=1).sum()))


def score_trial(folder, obj, condition, *, recompute=False):
    """Validate the physical condition and recompute PARK's signed surface and 100 Hz coverage."""
    import torch
    from PARK.core.common import write
    from PARK.core.study import archived
    from PARK.evaluation import metrics

    from Module.wrist_active.evaluate import Reference, reconstruct

    torch.set_num_threads(1)
    folder = Path(folder)
    release = read(base.ROOT / "PARK/release.json")
    fixture_path = base.FIXTURES / "fixtures.json"
    if digest(fixture_path) != release["runtime_dependencies"][fixture_path.relative_to(base.ROOT).as_posix()]:
        raise ValueError("Unmatched formal paired fixtures")
    fixture = read(fixture_path)
    spec = fixture["objects"][obj]
    _, original, official_scores, official_run = archived(obj, base.SECONDS, spec["seed"], "AUDE")
    run = read(folder / "run.json")
    validate_controller(run, obj, condition, spec["seed"], fixture["checkpoint"])
    for field, filename in (("model_sha256", "model.mjb"), ("offline_mesh_sha256", "surface.obj")):
        if not run[field] == spec[filename] == digest(base.FIXTURES / obj / filename):
            raise ValueError(f"Unmatched paired {filename} for {obj}")
    start = base.FIXTURES / obj / "initial.npz"
    with np.load(start, allow_pickle=False) as initial:
        initial_id = sha256(initial["integration_state"].tobytes()).hexdigest()
    if (spec["start"] != "start-02" or digest(start) != spec["initial.npz"]
            or not initial_id == spec["integration_state"] == run["initial_sha256"] == official_run["initial_sha256"]):
        raise ValueError(f"Unmatched paired starting state for {obj}")
    sig, official_sig = run["signature"], official_run["signature"]
    if (sig["source"] != official_sig["source"] or
            any(sig["code"].get(k) != v for k, v in official_sig["code"].items())):
        raise ValueError(f"Different upstream AUDE planner in {folder}")
    for name, fingerprint in run["hashes"].items():
        if Path(name).name != name or digest(folder / name) != fingerprint:
            raise ValueError(f"Raw file checksum mismatch at {folder}/{name}")
    identity = dict(object=obj, method=condition, seed=spec["seed"], trace_sha256=run["trace_sha256"],
                    plans_sha256=run["hashes"]["plans.jsonl"], initial_sha256=run["initial_sha256"],
                    model_sha256=run["model_sha256"], reference_sha256=run["offline_mesh_sha256"],
                    evaluator_sha256=official_scores["signature"]["metrics"],
                    scoring_gpis_sha256=official_scores["signature"]["gpis"],
                    report_code_sha256=digest(Path(__file__)), official_aude_trace_sha256=official_run["trace_sha256"],
                    run_sha256=digest(folder / "run.json"), archive_folder=f"{obj}/{condition}")
    if condition == "aude_height":
        identity["height_audit_code_sha256"] = digest(Path(__file__).with_name("height_evidence.py"))
    plans = [json.loads(line) for line in (folder / "plans.jsonl").read_bytes().splitlines()]
    with np.load(folder / "trace.npz", allow_pickle=False) as tr, np.load(original / "trace.npz",
                                                                         allow_pickle=False) as ref_trace:
        for key in INITIAL_FIELDS:
            np.testing.assert_array_equal(tr[key][0], ref_trace[key][0], err_msg=f"{obj}/{condition}/{key}")
        extras = condition_evidence(run, tr, plans)
        valid, load = base.qualified(tr)
        ticks, fingers = np.nonzero(valid)
        points, normals = tr["points_m"][ticks, fingers], tr["normals_outward"][ticks, fingers]
        qpos = tr["qpos"]
        np.testing.assert_allclose(run["wall_s"], tr["wall_s"][-1], atol=1e-9, rtol=0)
    extras.update(base.contacts_and_motion(qpos, valid, load))
    extras.update(base.acquisition_times(run, plans, folder / "timing.npz"))
    receipt = folder / "score.json"
    if receipt.is_file() and not recompute:
        current = read(receipt)
        if (all(current.get(k) == v for k, v in {**identity, **extras}.items()) and
                all((folder / name).is_file() and digest(folder / name) == current.get(key)
                    for name, key in ((base.ENDPOINT, "endpoint_sha256"), ("coverage.npz", "coverage_sha256")))):
            return current
    meta = dict(method=condition, object=obj, seconds=base.SECONDS, seed=spec["seed"], source=".",
                trace_sha256=run["trace_sha256"], initial_sha256=run["initial_sha256"], model_sha256=run["model_sha256"])
    with TemporaryDirectory(dir=folder) as temp:
        stage = Path(temp)
        canonical = metrics.score(folder, stage, meta, base.EVALUATOR)
        final = canonical["final"]
        ref = Reference(base.FIXTURES / obj / "surface.obj")
        curves, registered = ref.area(points, normals, ticks, base.TICKS)
        ref.neighbors.cache_clear()
        with (stage / "coverage-motion.csv").open(encoding="utf-8", newline="") as stream:
            checkpoints = list(csv.DictReader(stream))
        for curve, key in zip(curves, ("c10", "c25"), strict=True):
            np.testing.assert_allclose(curve[::base.HZ], [float(r[key]) for r in checkpoints], rtol=0, atol=1e-13)
        rebuilt = stage / "pre-encoding.ply"
        endpoint = reconstruct(points, normals, ref, rebuilt, "signed", signed_device="cuda")
        if (canonical["registered"] != registered or endpoint["status"] != "ok"
                or digest(rebuilt) != digest(stage / base.ENDPOINT)):
            raise ValueError(f"Failed to reproduce the scored canonical mesh for {obj}/{condition}")
        np.testing.assert_allclose([endpoint["fscore"], endpoint["mean_error_mm"]], [final["f5"], final["surface_error_mm"]],
                                   rtol=0, atol=1e-12)
        np.savez_compressed(stage / "coverage.npz", time_s=np.arange(base.TICKS + 1) / base.HZ,
                            c10=curves[0], c25=curves[1])
        score = dict(**identity, step=None, **{k: final[k] for k in ("c25", "c10", "f5", "surface_error_mm", "chamfer_mm2")},
                     **extras, registered=registered, precision5=endpoint["precision"], recall5=endpoint["recall"],
                     gpis_score_s=endpoint["compute_s"], endpoint_sha256=digest(rebuilt),
                     coverage_sha256=digest(stage / "coverage.npz"))
        write(stage / "score.json", score)
        for name in (base.ENDPOINT, "coverage.npz", "score.json"):
            (stage / name).replace(folder / name)
    return score


def summarize(rows):
    from . import height_evidence

    old = [r for r in rows if r["method"] in base.CONDITIONS]
    result = base.summarize(old, require_original=True)
    indexed = {(r["object"], r["method"]): r for r in rows}
    expected = {(obj, c) for obj in base.OBJECTS for c in CONDITIONS}
    if len(rows) != len(expected) or set(indexed) != expected or len({r["trace_sha256"] for r in rows}) != len(expected):
        raise ValueError("The extended comparison requires 25 distinct acquisitions")
    result["conditions"] = CONDITIONS
    result["runs"] = len(rows)
    result["condition_evidence"] = {}
    cal = read(height_evidence.CALIBRATION)
    result["height_protocol"] = dict(axis_palm=cal["axis_palm"], frame=cal["frame"],
                                     calibration_sha256=digest(height_evidence.CALIBRATION),
                                     lower_m=cal["lower_m"], upper_m=cal["upper_m"], training_median_m=cal["target_m"],
                                     quantiles=cal["probabilities"], source_weight=cal["source_weight"],
                                     source="Module/active_2/height_calibration.json",
                                     statistic="Median signed distance from wrist to qualified contact(s) along palm axis.",
                                     all_frames="24,000 post-initial measured wrist poses with the latest qualified contact patch.",
                                     fresh_frames="Post-initial frames containing a current qualified contact.")
    for obj in base.OBJECTS:
        reference = indexed[obj, "planned"]
        evidence = result["condition_evidence"][obj] = {}
        for condition in ADDITIONAL:
            row = indexed[obj, condition]
            if any(row[k] != reference[k] for k in PAIR_FIELDS):
                raise ValueError(f"Mismatched pairing or evaluator at {obj}/{condition}")
            result["objects"][obj]["metrics"][condition] = {k: row[k] for k in base.METRICS}
            extras = {k: v for k, v in row.items() if k not in indexed[obj, "compliance"]}
            evidence[condition] = dict(trace_sha256=row["trace_sha256"], endpoint_sha256=row["endpoint_sha256"], **extras)
        evidence["planned"] = {k: v for k, v in reference.items()
                               if k.startswith("measured_") or k in ("trace_sha256", "height_calibration_sha256")}
        for condition in ("planned", "aude_height"):
            row = indexed[obj, condition]
            if row["height_calibration_sha256"] != result["height_protocol"]["calibration_sha256"]:
                raise ValueError(f"Different height statistics calibration for {obj}/{condition}")
            result["objects"][obj]["metrics"][condition].update({k: row[k] for k in height_evidence.RATIOS})
    for condition in ADDITIONAL:
        means = result["macro_mean"][condition] = {}
        for field in base.METRICS:
            values = [indexed[o, condition][field] for o in base.OBJECTS]
            means[field] = float(np.mean(values)) if None not in values else None
    for condition in ("planned", "aude_height"):
        for key in height_evidence.RATIOS:
            values = [indexed[o, condition][key] for o in base.OBJECTS]
            result["macro_mean"][condition][key] = float(np.mean(values)) if None not in values else None
    return result


def main():
    from PARK.core.common import table, write

    from .height_evidence import add_baseline_height

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_OUT, help="Additional data root with runs/ and report/")
    parser.add_argument("--base", type=Path, default=base.ROOT_OUT, help="Original three-condition data root")
    parser.add_argument("--object", choices=base.OBJECTS)
    parser.add_argument("--condition", choices=ADDITIONAL)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    release = read(base.ROOT / "PARK/release.json")
    checkpoint = read(base.ROOT / "PARK/contract.json")["finger_policy"]["checkpoint_sha256"]
    rows = []
    for obj in (args.object,) if args.object else base.OBJECTS:
        if not args.object and not args.condition:
            for c in base.CONDITIONS:
                folder = args.base / "runs" / obj / c
                original, _ = verify_trial(folder, obj, c, release, checkpoint)
                if c == "planned":
                    original = add_baseline_height(original, folder)
                rows.append(original)
        for c in (args.condition,) if args.condition else ADDITIONAL:
            s = score_trial(args.root / "runs" / obj / c, obj, c, recompute=args.recompute)
            rows.append(s)
            print(f"{obj}/{c} C25 {s['c25']:.4%}, F@5 {s['f5']:.4%}", flush=True)
    if not args.object and not args.condition:
        result = summarize(rows)
        index = ("object", "method", "seed", "step", *base.METRICS)
        table(args.root / "report/trials.csv", [{**{k: r[k] for k in index}, **r} for r in rows])
        write(args.root / "report/results.json", result)


if __name__ == "__main__":
    main()
