"""Run the retained palm controller with an explicit cracker-only timing profile."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from unittest.mock import patch

from . import palm_mesh_recontact as mesh
from . import palm_progress_recontact as retained
from .experiment import save_json


@dataclass(frozen=True)
class Timing:
    single_stall_ticks: int
    retry_ticks: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.single_stall_ticks, self.retry_ticks)):
            raise ValueError("Both recontact timings must be positive integral 100 Hz ticks")


DEFAULT = Timing(retained.SINGLE_STALL_TICKS, mesh.RETRY_TICKS)
PROFILE = Path(__file__).with_name("retained_object_parameters.json")


def release(timing):
    if timing == DEFAULT:
        return retained.RELEASE
    return f"{retained.RELEASE}-cracker-single{timing.single_stall_ticks}-retry{timing.retry_ticks}"


def run(folder, obj, seconds=240, device="cuda", *, cracker_timing=None):
    timing = DEFAULT
    if obj == "cracker":
        timing = cracker_timing or Timing(**json.loads(PROFILE.read_bytes())["cracker"])
    if timing == DEFAULT:
        out = retained.run(folder, obj, seconds, device)
    else:
        # The native stack runs one controller per process. Scope the existing
        # scalar lookups to that rollout, including its worker, and sign their
        # values into the native release so cached runs cannot cross profiles.
        native = retained.tactile_field.run
        def signed(*args, source_files, release, **kwargs):
            if release != retained.RELEASE:
                raise ValueError("Unexpected base controller")
            return native(*args, source_files=(*source_files, __file__), release=profile_release, **kwargs)
        profile_release = release(timing)
        with (patch.object(retained, "SINGLE_STALL_TICKS", timing.single_stall_ticks),
              patch.object(mesh, "RETRY_TICKS", timing.retry_ticks),
              patch.object(retained.tactile_field, "run", signed)):
            out = retained.run(folder, obj, seconds, device)
    record = dict(object=obj, hz=100, zero_ticks=retained.ZERO_TICKS, **asdict(timing),
                  release=release(timing), base=retained.RELEASE)
    dest = Path(out) / "active-parameters.json"
    if dest.exists() and json.loads(dest.read_bytes()) != record:
        raise ValueError(f"Parameter receipt disagrees with completed run at {out}")
    if not dest.exists():
        save_json(dest, record)
    return out


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--object", choices=tuple(mesh.palm.OBJECTS), required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--seconds", type=int, default=240)
    cli.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = cli.parse_args()
    print(run(args.out, args.object, args.seconds, args.device), flush=True)
