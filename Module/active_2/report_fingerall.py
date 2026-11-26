"""Evaluate the wrist height intervention with the complete frozen single_all + MCC hand."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PARK.identity import digest, read

from . import height_evidence, verify_extended
from . import report as base
from . import report_extended as extended
from .verify import checked_bytes

COMBINED = extended.MCC_HEIGHT
CONTROL = "aude_mcc"
PRIMARY = (*base.CONDITIONS, CONTROL, COMBINED)
AUXILIARY = ("aude_height",)
CONDITIONS = {**base.CONDITIONS, CONTROL: extended.ADDITIONAL[CONTROL],
              COMBINED: "AUDE 腕规划 + single_all/MCC + 掌轴距离约束",
              **{k: extended.ADDITIONAL[k] for k in AUXILIARY}}
ROOT_OUT = base.ROOT_OUT / "fingerall"
PUBLISHED_CONTROL_DRIVER = "8064f6ea0f8a46835784385ae9571c3b0c1fdf35906e02e5bd50e2f1ec11584f"


def score_trial(folder, obj, condition, *, recompute=False):
    if condition != COMBINED:
        raise ValueError(f"Expected {COMBINED}, got {condition}")
    return extended.score_trial(folder, obj, condition, recompute=recompute)


def paired_evidence(control_dir, treated_dir, control, treated):
    """Compare physical states, unexecuted AUDE finger solutions, and the hand before the first intervention."""
    if any(control[k] != treated[k] for k in extended.PAIR_FIELDS):
        raise ValueError("The two complete-hand runs use different fixtures, seed or evaluator")
    raw = [json.loads(checked_bytes(d / "run.json", s["run_sha256"]))
           for d, s in ((control_dir, control), (treated_dir, treated))]
    control_run, treated_run = raw
    policy = read(base.ROOT / "PARK/contract.json")["finger_policy"]
    for key in (*policy, "inference_seed", "bootstrap_frames", "queries", "handoff_misses", "skipped_queries"):
        if control_run["policy"][key] != treated_run["policy"][key]:
            raise ValueError(f"Different lower finger policy setting: {key}")
    for k in ("actuation", "joint_limits_rad", "robot_sha256", "initial_file_sha256"):
        if control_run[k] != treated_run[k]:
            raise ValueError(f"Different LEAP motor or physical hand: {k}")
    original = control_run["signature"]
    modified = treated_run["signature"]
    driver = Path(__file__).with_name("mcc.py")
    if original["code"].get(driver.name) not in (PUBLISHED_CONTROL_DRIVER, digest(driver)):
        raise ValueError("The unconstrained hand was recorded with an unrecognized acquisition driver")
    for name in ("mcc.py", "height.py", "fingerall_height.py"):
        if modified["code"].get(name) != digest(driver.with_name(name)):
            raise ValueError(f"The constrained hand was recorded with a different {name}")
    for k in ("source", "execute_planned_fingers", "finger_mcc", "wrist_mcc", "wrist_control", "history"):
        if original[k] != modified[k]:
            raise ValueError(f"Different upstream/lower control setting: {k}")
    # mcc.py adds the optional projection. The AUDE solver, finger policy, MCC, and other dependencies are shared.
    for filename, fingerprint in original["code"].items():
        if filename != "mcc.py" and modified["code"].get(filename) != fingerprint:
            raise ValueError(f"Different shared controller source: {filename}")
    plans = [[json.loads(line) for line in checked_bytes(d / "plans.jsonl", s["plans_sha256"]).splitlines()]
             for d, s in ((control_dir, control), (treated_dir, treated))]
    ticks = [p["tick"] for p in plans[1] if p["height_correction_m"] > 1e-10]
    first = ticks[0] if ticks else None
    until = base.TICKS if first is None else first
    for reference, candidate in zip(*plans, strict=True):
        if candidate["tick"] > until:
            break
        np.testing.assert_allclose(reference["wrist"], candidate["raw_wrist"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(reference["aude_joints_unexecuted_rad"], candidate["raw_joints_rad"],
                                   rtol=0, atol=1e-10)
    fields = (*extended.INITIAL_FIELDS, "wrist_mocap", "goal_wrist", "finger_nominal_rad",
              "mcc_bias_rad", "finger_command_rad", "policy_wrist_preview")
    prefix_rows = until + 1  # A plan at tick t first affects the logged state at t+1.
    with np.load(io.BytesIO(checked_bytes(control_dir / "trace.npz", control["trace_sha256"])),
                 allow_pickle=False) as old, np.load(io.BytesIO(checked_bytes(
                     treated_dir / "trace.npz", treated["trace_sha256"])), allow_pickle=False) as new:
        exact = True
        for key in fields:
            a, b = old[key][:prefix_rows], new[key][:prefix_rows]
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-10, err_msg=f"{treated['object']}: {key} before projection")
            exact = exact and np.array_equal(a, b, equal_nan=True)
    return dict(primary_control_trace_sha256=control["trace_sha256"], first_projection_tick=first,
                primary_control_driver_sha256=original["code"]["mcc.py"],
                constrained_driver_sha256=modified["code"]["mcc.py"],
                primary_control_prefix_frames=prefix_rows, primary_control_prefix_exact=bool(exact))


def summarize(rows):
    old = [r for r in rows if r["method"] != COMBINED]
    result = extended.summarize(old)
    index = {(r["object"], r["method"]): r for r in rows}
    keys = {(obj, method) for obj in base.OBJECTS for method in CONDITIONS}
    if set(index) != keys or len(rows) != len(keys) or len({r["trace_sha256"] for r in rows}) != len(keys):
        raise ValueError("The complete hand comparison requires 30 unique acquisitions")
    result.update(conditions=CONDITIONS, primary_conditions=list(PRIMARY), auxiliary_conditions=list(AUXILIARY),
                  runs=len(rows))
    result["height_protocol"].update(primary_comparison=[CONTROL, COMBINED],
                                     auxiliary_comparison=["planned", "aude_height"])
    for obj in base.OBJECTS:
        control = index[obj, CONTROL]
        treated = index[obj, COMBINED]
        if (any(control[k] != treated[k] for k in extended.PAIR_FIELDS) or
                treated["primary_control_trace_sha256"] != control["trace_sha256"] or
                not control["height_calibration_sha256"] == treated["height_calibration_sha256"]
                == result["height_protocol"]["calibration_sha256"]):
            raise ValueError(f"Unmatched direct control or calibration for {obj}")
        public = result["objects"][obj]["metrics"]
        public[COMBINED] = {k: treated[k] for k in base.METRICS}
        for k in extended.MCC_CONDITIONS:
            public[k].update({field: index[obj, k][field] for field in height_evidence.RATIOS})
        result["objects"][obj]["metrics"] = {k: public[k] for k in CONDITIONS}
        baseline_keys = set(index[obj, "compliance"])
        result["condition_evidence"][obj][COMBINED] = {
            k: v for k, v in treated.items() if k not in baseline_keys or k in ("trace_sha256", "endpoint_sha256")}
    result["macro_mean"][COMBINED] = {}
    for condition in extended.MCC_CONDITIONS:
        fields = (*base.METRICS, *height_evidence.RATIOS) if condition == COMBINED else height_evidence.RATIOS
        for field in fields:
            values = [index[o, condition][field] for o in base.OBJECTS]
            result["macro_mean"][condition][field] = float(np.mean(values)) if None not in values else None
    result["macro_mean"] = {k: result["macro_mean"][k] for k in CONDITIONS}
    return result


def verified_rows(original, additional, new, *, geometry=False):
    old, _ = verify_extended.verify_all(original, additional, geometry=geometry)
    index = {(r["object"], r["method"]): r for r in old}
    release = read(base.ROOT / "PARK/release.json")
    checkpoint = read(base.ROOT / "PARK/contract.json")["finger_policy"]["checkpoint_sha256"]
    for obj in base.OBJECTS:
        control_dir = additional / obj / CONTROL
        treated_dir = new / obj / COMBINED
        control = height_evidence.add_baseline_height(index[obj, CONTROL], control_dir)
        treated, _ = verify_extended.verify_additional_trial(treated_dir, obj, COMBINED, release, checkpoint,
                                                            geometry=geometry)
        index[obj, CONTROL] = control
        index[obj, COMBINED] = {**treated, **paired_evidence(control_dir, treated_dir, control, treated)}
    return [index[obj, condition] for obj in base.OBJECTS for condition in CONDITIONS]


def main():
    from PARK.core.common import table, write

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_OUT, help="New data root with runs/ and report/")
    parser.add_argument("--base", type=Path, default=base.ROOT_OUT, help="Original 15-run data root")
    parser.add_argument("--extended", type=Path, default=extended.ROOT_OUT, help="Prior 10-run data root")
    parser.add_argument("--object", choices=base.OBJECTS, help="Score one new run without assembling the report")
    parser.add_argument("--recompute", action="store_true", help="Re-score the five new acquisitions")
    args = parser.parse_args()
    for obj in (args.object,) if args.object else base.OBJECTS:
        s = score_trial(args.root / "runs" / obj / COMBINED, obj, COMBINED, recompute=args.recompute)
        print(f"{obj}/{COMBINED} C25 {s['c25']:.4%}, F@5 {s['f5']:.4%}", flush=True)
    if not args.object:
        rows = verified_rows(args.base / "runs", args.extended / "runs", args.root / "runs")
        result = summarize(rows)
        index = ("object", "method", "seed", "step", *base.METRICS)
        table(args.root / "report/trials.csv", [{**{k: r[k] for k in index}, **r} for r in rows])
        write(args.root / "report/results.json", result)


if __name__ == "__main__":
    main()
