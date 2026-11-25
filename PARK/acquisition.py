"""Run PARK on the shared physical controller and record the programmed and measured wrist."""
from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from PARK.core.common import configuration, digest, planning, read, rot, write
from PARK.identity import verify_identity

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = read(HERE / "config/acquisition.json")
MODELS = ROOT / CONFIG["models"]


def acquire(obj, seconds, repeat, dest, smoke=False):
    from Module.palmreach6.__main__ import FIXTURES, ROOT as native_root
    from Module.wrist_active import palm_exploration, palm_progress_recontact, policy_reach, tactile_field
    from PARK.evaluation.native import run_baseline
    from PARK.measured_start import MeasuredStartWrist, MeasuredStartWorker
    from PARK.core.study import check_run
    from PARK.identity import verify_files

    release = read(HERE / "release.json")
    verify_files(ROOT, release["planner_inputs"])
    verify_files(native_root, release["controller_inputs"])
    contract = read(HERE / "contract.json")["original_files"]
    if any(release["controller_inputs"].get(name) != value for name, value in contract.items()):
        raise ValueError("Inconsistent native execution contract")
    frontier = read(HERE / "config/frontier.json")
    seed = read(FIXTURES / "fixtures.json")["objects"][obj]["seed"] + repeat
    signature = dict(protocol_version=1, method="PARK", variant=CONFIG["variant"], object=obj,
                     seconds=seconds, seed=seed, files=release["planner_inputs"], native=release["controller_inputs"])
    if dest.exists():
        receipt = read(dest / "adapter.json")
        verify_identity(receipt, dest, HERE / "release.json")
        if any(receipt["signature"][key] != signature[key] for key in ("object", "seconds", "seed", "variant")):
            raise ValueError(f"Different run already exists at {dest}")
        check_run(dest, obj, seconds, seed, "PARK")
        verify_files(dest, receipt["telemetry"])
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp)
        native = stage / "native"
        global_log = stage / "global-plans.jsonl"
        raw_run = tactile_field.run
        cfg = planning(configuration(ROOT / CONFIG["planner"]))
        with global_log.open("w", encoding="utf-8") as events, ExitStack() as scope:
            record = SimpleNamespace(nominal=[], events=events)
            def integrated(*args, wrist_cls, worker_cls, source_files, release, **kwargs):
                if wrist_cls.func is not palm_progress_recontact.ProgressWrist or worker_cls is not palm_progress_recontact.ProgressWorker:
                    raise ValueError("Unexpected native controller entry point")
                controller = partial(MeasuredStartWrist, models=MODELS, cfg=cfg, variant=CONFIG["variant"], recorder=record, frontier=frontier)
                return raw_run(*args, wrist_cls=controller, worker_cls=MeasuredStartWorker,
                               source_files=(*source_files, HERE / "planner_adapter.py", HERE / "config/acquisition.json",
                                             HERE / "frontier_planner.py", HERE / "config/frontier.json", HERE / "measured_start.py"),
                               release="PARK", **kwargs)
            scope.enter_context(patch.object(tactile_field, "run", integrated))
            if smoke:
                # Tripwires cover both the ordinary selection and a fallback from mesh recontact.
                def forbidden(*args, **kwargs):
                    raise AssertionError("PalmReach-6 global action selection was invoked")
                for owner, key in ((palm_exploration, "palm_headings"), (policy_reach.PolicyReach, "predict"),
                                   (palm_exploration, "joint_gain")):
                    scope.enter_context(patch.object(owner, key, forbidden))
            # Reuse the frozen fixture, cracker profile and both original seed
            # injection points. The hook above is reached before constructing a wrist.
            run_baseline(obj, "palmreach-6", seconds, native, seed)
        run = check_run(native, obj, seconds, seed, CONFIG["method"])
        clock = read(native / "clock.json")
        if clock["release"] != "PARK-cuda" or not record.nominal:
            raise ValueError("PARK did not execute")
        clock.update(release="PARK", device="cuda")
        write(native / "clock.json", clock)
        nominal = np.stack(record.nominal)
        with np.load(native / "trace.npz", allow_pickle=False) as trace:
            actual, command = trace["qpos"][:, :7], trace["palm_command"]
        if nominal.shape != actual.shape or not np.isfinite(nominal).all():
            raise ValueError("Incomplete nominal wrist telemetry")
        translation = np.linalg.norm(nominal[:, :3] - command[:, :3], axis=1)
        angle = (rot(command) * rot(nominal).inv()).magnitude()
        np.savez_compressed(native / "wrist.npz", nominal=nominal, corrected_command=command, actual=actual,
                            nominal_command_distance_m=translation, nominal_command_angle_rad=angle)
        global_log.rename(native / global_log.name)
        rows = [json.loads(line) for line in (native / "global-plans.jsonl").read_bytes().splitlines()]
        with (native / "plans.jsonl").open(encoding="utf-8") as stream:
            fitted = sum("palm_height_fit_mm" in json.loads(line) for line in stream)
        accepted = sum(row["reason"] == "new_target" for row in rows)
        if smoke and (not accepted or not fitted):
            raise ValueError("Smoke trial did not exercise a selected target and native palm support fit")
        write(native / "adapter.json", dict(acquisition_origin="native", signature=signature, entrypoint_sha256=digest(Path(__file__)),
              trace_sha256=run["trace_sha256"],
              telemetry={n: digest(native / n) for n in ("global-plans.jsonl", "wrist.npz")},
              smoke_tripwires=smoke, decisions=len(rows), new_targets=accepted, support_fits=fitted))
        native.rename(dest)


def trial_dir(root, seconds, obj, repeat):
    from Module.palmreach6.__main__ import FIXTURES
    seed = read(FIXTURES / "fixtures.json")["objects"][obj]["seed"] + repeat
    return root / "trials" / f"{seconds}s" / obj / f"seed-{seed}"
