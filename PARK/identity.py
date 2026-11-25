"""Content and source identity checks for deployed and published PARK trials."""
from hashlib import sha256
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_files(root, checksums):
    for name, expected in checksums.items():
        path = root / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"PARK checksum mismatch: {path}")


def verify_identity(receipt, native, manifest_path):
    manifest = read(manifest_path)
    sig = receipt["signature"]
    if sig["method"] != manifest["name"] or sig["native"] != manifest["controller_inputs"]:
        raise ValueError(f"Unexpected PARK controller: {native}")
    origin = receipt["acquisition_origin"]
    if origin == "retained":
        # Original source bytes and normalized public artifacts are independently pinned.
        trial = next((t for t in manifest["published_runs"] if
                      (t["object"], t["seconds"], t["seed"]) == (sig["object"], sig["seconds"], sig["seed"])), None)
        if trial is None or digest(native / "adapter.json") != trial["files"]["native/adapter.json"]:
            raise ValueError(f"Unrecognized published acquisition: {native}")
        expected = manifest["published_acquisition_inputs"]
    elif origin == "native":
        expected = manifest["planner_inputs"]
        if receipt["entrypoint_sha256"] != manifest["workspace_dependencies"]["PARK/acquisition.py"]:
            raise ValueError(f"Different PARK acquisition implementation: {native}")
    else:
        raise ValueError(f"Unknown PARK acquisition origin at {native}")
    if sig["files"] != expected:
        raise ValueError(f"Different PARK planning inputs at {native}")
