"""Check physical recordings and locate matched reference scores."""
from pathlib import Path

from Module.palmreach6.__main__ import FIXTURES
from .common import ROOT, digest, read

METHODS = {"AUDE": "aude", "PalmReach-6": "palmreach-6"}


def check_run(source, obj, seconds, seed, method):
    if method not in ("PARK", *METHODS):
        raise ValueError(f"Unknown method: {method}")
    run = read(source / "run.json")
    fixture = read(FIXTURES / "fixtures.json")
    objspec = fixture["objects"][obj]
    sig = run["signature"]
    if (sig.get("obj", sig.get("object")) != obj or sig["seconds"] != seconds or sig["seed"] != seed
            or sig.get("paced", False) or run["steps"] != 100*seconds or run["seed"] != seed
            or run["initial_sha256"] != objspec["integration_state"] or run["model_sha256"] != objspec["model.mjb"]
            or run["offline_mesh_sha256"] != objspec["surface.obj"]):
        raise ValueError(f"Unmatched paired trial at {source}")
    if method == "AUDE":
        if sig["planner_feedback"] != "physical" or sig["period"] != 90 or sig["wrist_mcc"] or sig["finger_mcc"]:
            raise ValueError(f"Incorrect AUDE controller at {source}")
    else:
        p = run["policy"]
        if (sig["start"] != objspec["start"] or run["initial_file_sha256"] != objspec["initial.npz"]
                or run["robot_sha256"] != fixture["hand_sha256"] or p["checkpoint_sha256"] != fixture["checkpoint"]
                or p["inference_seed"] != seed or p["schedule"] != "100ms chunks, 50ms buffered handoff"
                or p["action_points"] != 16):
            raise ValueError(f"Unmatched deployed controller at {source}")
        if method == "PalmReach-6" and "palm_progress_recontact.py" not in sig["code"]:
            raise ValueError(f"Unmatched PalmReach-6 controller at {source}")
    if (digest(source / "trace.npz") != run["trace_sha256"] or digest(source / "plans.jsonl") !=
            (run.get("plan_sha256") or run["hashes"]["plans.jsonl"])):
        raise ValueError(f"Raw trial checksum mismatch at {source}")
    return run


def archived(obj, seconds, seed, method):
    from PARK.evaluation import metrics
    key = METHODS[method]
    path = ROOT/"outputs/PARK/experiments"/f"{seconds}s"/obj/key/f"seed-{seed}"
    meta, score = read(path/"trial.json"), read(path/"metrics.json")
    source = (path/meta["source"]).resolve()
    run = check_run(source, obj, seconds, seed, method)
    if (meta != score["meta"] or meta["method"] != key or meta["object"] != obj or meta["seed"] != seed
            or meta["seconds"] != seconds or meta["trace_sha256"] != run["trace_sha256"]
            or score["signature"]["trace"] != run["trace_sha256"] or score["signature"]["plans"] != digest(source/"plans.jsonl")
            or score["signature"]["metrics"] != digest(Path(metrics.__file__))
            or score["signature"]["gpis"] != digest(ROOT/"outputs/PARK/models/evaluator/gpis.pt")):
        raise ValueError(f"Unmatched historical score at {path}")
    return path, source, score, run
