"""Score PARK and compare it to the two signed historical acquisitions."""
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from Module.palmreach6.__main__ import FIXTURES
from PARK.evaluation import metrics
from PARK.identity import verify_identity
from PARK.core.common import digest, read, table, write
from PARK.core.study import archived, check_run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = read(HERE / "config/acquisition.json")
METHODS = ("PARK", "PalmReach-6", "AUDE")


def verify_adapter(native, obj, seconds, seed, run):
    receipt = read(native / "adapter.json")
    sig = receipt["signature"]
    if (sig["method"], sig["variant"], sig["object"], sig["seconds"], sig["seed"]) != (
            METHODS[0], CONFIG["variant"], obj, seconds, seed):
        raise ValueError(f"Wrong planner or experimental cell at {native}")
    if receipt["trace_sha256"] != run["trace_sha256"] or any(
            digest(native / key) != sha for key, sha in receipt["telemetry"].items()):
        raise ValueError(f"Incorrect physical trace or telemetry at {native}")
    verify_identity(receipt, native, ROOT / "PARK/release.json")
    signed = run["signature"]["code"]
    for key in ("palm_support.py", "palm_progress_recontact.py", "palm_mesh_recontact.py",
                "buffered_fingers.py", "sweep_experiment.py"):
        if signed[key] != sig["native"]["Module/wrist_active/" + key]:
            raise ValueError(f"Incorrect native {key} at {native}")
    return receipt


def score_trial(folder):
    import torch
    torch.set_num_threads(1)
    native = folder / "native"
    receipt = read(native / "adapter.json")
    sig = receipt["signature"]
    obj, seconds, seed = (sig[k] for k in ("object", "seconds", "seed"))
    run = check_run(native, obj, seconds, seed, METHODS[0])
    verify_adapter(native, obj, seconds, seed, run)
    out = folder / "score"
    out.mkdir(exist_ok=True)
    first = read(FIXTURES / "fixtures.json")["objects"][obj]["seed"]
    meta = dict(method=METHODS[0], object=obj, seconds=seconds, seed=seed, repeat=seed-first,
                source=Path(os.path.relpath(native, out)).as_posix(), archived=False, trace_sha256=run["trace_sha256"],
                initial_sha256=run["initial_sha256"], model_sha256=run["model_sha256"])
    metrics.score(native, out, meta, ROOT / "outputs/PARK/models/evaluator")
    trial_file = out / "trial.json"
    if not trial_file.is_file() or read(trial_file) != meta:
        write(trial_file, meta)


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def summarize(a):
    if not len(a):
        return dict(n=0, mean=None, sd=None)
    return dict(n=len(a), mean=float(np.mean(a)), sd=float(np.std(a, ddof=1)) if len(a) > 1 else None)


def low_gain_windows(motion):
    checkpoints = [p for p in motion if float(p["seconds"]) % 10 == 0]
    stalled = np.diff([float(p["c25"]) for p in checkpoints]) < .001
    path = np.diff([float(p["path_m"]) for p in checkpoints])
    return stalled, path


def plateau_metrics(motion):
    stalled, path = low_gain_windows(motion)
    edges = np.flatnonzero(np.diff(np.r_[False,stalled,False]))
    return dict(stalled_fraction=float(stalled.mean()),
                longest_stall_s=int(10*(edges[1::2]-edges[::2]).max(initial=0)),
                stalled_path_m=float(path[stalled].sum()))


def latency(row, prefix, samples):
    row[prefix+"_n"] = len(samples)
    row[prefix+"_mean_ms"] = float(np.mean(samples)) if len(samples) else None
    row[prefix+"_p95_ms"] = float(np.percentile(samples, 95)) if len(samples) else None


def planning_data(native, method):
    local, selection, global_scope = [], [], []
    cycles, observation_to_step, aude_directions, aude_ik = [], [], [], []
    ticks, applied = [], []
    with np.load(native / "trace.npz", allow_pickle=False) as trace:
        wall = trace["wall_s"]
    height, turn = [], []
    new_targets = decisions = 0
    recovery_installs = geometry_replans = 0
    reasons = Counter()
    extra = {"gpis_update": [], "gpis_query": [], "kde": [], "candidate": [], "evaluation": []}
    frontier = {k: [] for k in ("footprint_overlap", "residence_s", "dwell_penalty",
                                "unproductive_pursuit_s", "failed_penalty", "global_route_distance_m")}
    for r in records(native / "plans.jsonl"):
        local.append(r["plan_ms" if method == "AUDE" else "planning_ms"])
        ticks.append(r["tick"])
        applied.append(r.get("apply_tick", r["tick"]))
        if method == "AUDE":
            decisions += 1
            cycles.append(local[-1])
            observation_to_step.append(1000 * (wall[r["tick"]+1] - wall[r["tick"]]))
            if r.get("gp_direction_ms", 0) > 0:
                aude_directions.append(r["gp_direction_ms"])
            if r.get("fit_calls", 0):
                aude_ik.append(r["ik_ms"])
            continue
        if r["local_period_s"] != .05:
            raise ValueError(f"Changed native support period in {native}")
        recovery_installs += int(r.get("mesh_recontact_installed", False))
        if "palm_height_fit_mm" in r:
            height.append(abs(r["palm_height_fit_mm"]))
            turn.append(np.linalg.norm(r["palm_support_turn_rad_s"]))
        if method == "PalmReach-6":
            event = r.get("global_event") or {}
            if "palm_selection_ms" in event:
                selection.append(event["palm_selection_ms"])
                global_scope.append(r["global_ms"])
                cycles.append(local[-1])
                observation_to_step.append(1000 * (wall[r["apply_tick"]+1] - wall[r["tick"]]))
                decisions += 1
                new_targets += int(event.get("direction_chosen", False))
                reasons[event.get("source", "unknown")] += 1
        elif (r.get("global_event") or {}).get("source") in ("palm_hold", "palm_reach_information"):
            raise ValueError(f"Unexpected historical global decision at {native}")
    if method != "AUDE" and not np.array_equal(applied, np.arange(0, len(wall)-1, 5)):
        raise ValueError(f"Missing or shifted local updates in {native}")
    if method == "PARK":
        submissions = np.asarray(ticks)
        for r in records(native / "global-plans.jsonl"):
            global_scope.append(r["planning_ms"])
            # Only count a pair when the following local submission is inside the budget.
            j = np.searchsorted(submissions, r["apply_tick"])
            if j < len(local):
                cycles.append(r["planning_ms"] + local[j])
                observation_to_step.append(1000 * (wall[applied[j]+1] - wall[r["tick"]]))
            decisions += 1
            new_targets += int(r["reason"] == "new_target")
            geometry_replans += int(r["reason"] == "new_target" and r.get("trigger") == "geometry")
            reasons[r["reason"]] += 1
            for k, samples in extra.items():
                if k+"_ms" in r:
                    samples.append(r[k+"_ms"])
            for k, samples in frontier.items():
                if k in r and r["reason"] in ("new_target", "keep_current_target"):
                    samples.append(r[k])
    row = dict(global_decisions=decisions, new_targets=new_targets if method != "AUDE" else None,
               accepted_geometry_replans=geometry_replans if method == METHODS[0] else None,
               mesh_recontact_installs=recovery_installs if method != "AUDE" else None,
               support_fit_events=len(height) if method != "AUDE" else None,
               support_abs_height_mm=float(np.mean(height)) if height else None,
               support_turn_rad_s=float(np.mean(turn)) if turn else None,
               nominal_command_mm=None, nominal_command_deg=None, command_actual_mm=None)
    for k, samples in frontier.items():
        row[k+"_mean"] = float(np.mean(samples)) if samples else None
        row[k+"_p95"] = float(np.percentile(samples, 95)) if samples else None
    row["scored_frontier_decisions"] = len(frontier["dwell_penalty"]) if frontier["dwell_penalty"] else None
    latency(row, "global", global_scope)
    latency(row, "six_direction_selection", selection)
    latency(row, "native_update", local)
    latency(row, "decision_and_local", cycles)
    latency(row, "observation_to_first_step", observation_to_step)
    latency(row, "aude_directions", aude_directions)
    latency(row, "aude_ik", aude_ik)
    finger_inference = []
    if method != "AUDE":
        with np.load(native / "timing.npz", allow_pickle=False) as timing:
            finger_inference = timing["inference_ms"]
    latency(row, "finger_inference", finger_inference)
    for k, samples in extra.items():
        latency(row, k, samples)
    # Sum separately measured CPU/GPU worker elapsed times, even if overlapped.
    # This is work per simulation second, not elapsed time or a deadline measure.
    row["measured_planning_work_ms"] = sum(local) + (sum(global_scope) if method == METHODS[0] else 0)
    if method == METHODS[0]:
        with np.load(native / "wrist.npz", allow_pickle=False) as wrist:
            row["nominal_command_mm"] = 1000 * float(wrist["nominal_command_distance_m"][1:].mean())
            row["nominal_command_deg"] = float(np.rad2deg(wrist["nominal_command_angle_rad"][1:]).mean())
            row["command_actual_mm"] = 1000 * float(np.linalg.norm(
                wrist["corrected_command"][1:, :3] - wrist["actual"][1:, :3], axis=1).mean())
    return row, reasons


def aggregate(rows, group_keys, columns):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in groups.items():
        for name in columns:
            values = [r[name] for r in group if r.get(name) is not None]
            result.append(dict(**dict(zip(group_keys, key)), metric=name, **summarize(values)))
    return result


def physical_start(native):
    with np.load(native / "trace.npz", allow_pickle=False) as tr:
        return {k: tr[k][0].copy() for k in ("qpos", "points_m", "normals_outward", "pad_valid",
                                             "pad_force_n", "region_load_n")}


def collect(root, budgets=None, *, objects=None, repeats=None):
    spec = read(FIXTURES / "fixtures.json")
    finger_settings = read(HERE / "contract.json")["finger_policy"]
    rows, curves, fingers, provenance = [], [], [], []
    objects = CONFIG["objects"] if objects is None else objects
    repeats = (0,) if repeats is None else repeats
    for seconds in CONFIG["budgets_s"] if budgets is None else budgets:
        for obj in objects:
            for repeat in repeats:
                seed = spec["objects"][obj]["seed"] + repeat
                initial = None
                recovery_config = None
                for method in METHODS:
                    if method == METHODS[0]:
                        folder = root / "trials" / f"{seconds}s" / obj / f"seed-{seed}"
                        scored, native = folder / "score", folder / "native"
                        run = check_run(native, obj, seconds, seed, method)
                        score = read(scored / "metrics.json")
                        receipt = verify_adapter(native, obj, seconds, seed, run)
                        if (score["meta"] != read(scored / "trial.json") or
                            (score["meta"]["method"], score["meta"]["object"], score["meta"]["seconds"],
                             score["meta"]["seed"]) != (method, obj, seconds, seed) or
                            score["signature"]["trace"] != run["trace_sha256"] or
                            score["signature"]["metrics"] != digest(Path(metrics.__file__)) or
                            score["signature"]["plans"] != run["plan_sha256"] or
                            score["signature"]["gpis"] != digest(ROOT / "outputs/PARK/models/evaluator/gpis.pt") or
                            not all((scored / name).is_file() for name in score["files"])):
                            raise ValueError(f"Unmatched evaluation or telemetry at {scored}")
                    else:
                        scored, native, score, run = archived(obj, seconds, seed, method)
                    start = physical_start(native)
                    if initial is None:
                        initial = start
                    else:
                        for k in start:
                            np.testing.assert_array_equal(start[k], initial[k], err_msg=f"{native}: {k}")
                    if method != "AUDE":
                        profile = read(native / "active-parameters.json")
                        if recovery_config is not None and profile != recovery_config:
                            raise ValueError(f"Different recovery configuration at {native}")
                        recovery_config = profile
                        for k, val in finger_settings.items():
                            if run["policy"][k] != val:
                                raise ValueError(f"Different finger policy {k} at {native}")
                    tags = dict(budget_s=seconds, object=obj, seed=seed, method=method)
                    timing, reasons = planning_data(native, method)
                    timing["planning_work_ms_per_sim_s"] = timing.pop("measured_planning_work_ms") / seconds
                    trial = dict(**tags, **score["final"], **timing)
                    provenance.append(dict(**tags, native=os.path.relpath(native, root),
                        score=os.path.relpath(scored / "metrics.json", root), trace=run["trace_sha256"],
                        initial=run["initial_sha256"], model=run["model_sha256"],
                        scoring_mesh=run["offline_mesh_sha256"], decision_reasons=dict(reasons),
                        executed_steps=run["steps"], missed_finger_handoffs=run.get("missed_handoffs"),
                        finger_underrun_ticks=run.get("underrun_ticks"), late_wrist_handoffs=run.get("late_wrist_handoffs")))
                    for finger, values in score["contact"].items():
                        fingers.append(dict(**tags, finger=finger, **values))
                    with (scored / "coverage-motion.csv").open(encoding="utf-8") as s:
                        motion = list(csv.DictReader(s))
                    quality = {p["seconds"]: p for p in score["checkpoints"]}
                    # Integrate only intervals bracketed by recorded causal checkpoints.
                    ts = np.array(list(quality), dtype=float)
                    trial["time_mean_f5_from_1s"] = float(np.trapz([p["f5"] for p in quality.values()], ts)
                                                           / (ts[-1] - ts[0]))
                    trial["time_mean_c25"] = float(np.trapz([float(p["c25"]) for p in motion],
                                                            [float(p["seconds"]) for p in motion]) / seconds)
                    trial["new_c25_pp_per_m"] = 100 * (trial["c25"] - float(motion[0]["c25"])) / trial["path_m"]
                    trial.update(plateau_metrics(motion))
                    trial["wall_s"] = run.get("wall_s")
                    if (native / "clock.json").exists():
                        trial["wall_s"] = read(native / "clock.json")["all_wall_s"]
                    rows.append(trial)
                    for p in motion:
                        t = float(p["seconds"])
                        curves.append(dict(**tags, seconds=t, f5=quality.get(t, {}).get("f5"),
                            surface_error_mm=quality.get(t, {}).get("surface_error_mm"),
                            chamfer_mm2=quality.get(t, {}).get("chamfer_mm2"),
                            **{k: float(p[k]) for k in ("c25", "c10", "path_m", "valid_samples", "new_voxels")}))
    return rows, curves, fingers, provenance
