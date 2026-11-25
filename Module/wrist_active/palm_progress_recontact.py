"""Six-direction palm exploration with mesh recontact after loss of tactile progress."""

import argparse
import json
from functools import partial
from pathlib import Path
import sys

from . import palm_mesh_recontact as mesh
from . import policy_reach, tactile_field
from .sweep import LOAD_LIMIT


ZERO_TICKS = 75
SINGLE_STALL_TICKS = 150
RELEASE = "palm-six-progress-mesh-recontact"


class ProgressWrist(mesh.MeshWrist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_ticks = self.single_ticks = 0

    def observe(self, data, touch, tick):
        super().observe(data, touch, tick)
        low_load = touch.loads.max() < LOAD_LIMIT
        pads = touch.measured_heads.sum()
        self.zero_ticks = self.zero_ticks + 1 if low_load and pads == 0 else 0
        self.single_ticks = self.single_ticks + 1 if low_load and pads == 1 else 0

    def trigger(self, tick):
        if self.zero_ticks >= ZERO_TICKS:
            return "sustained_zero"
        last_new = self.surface.last_observation_tick
        if self.single_ticks >= SINGLE_STALL_TICKS and last_new is not None and tick - last_new >= SINGLE_STALL_TICKS:
            return "single_without_new_cell"
        return None

    def can_recontact(self, touch, tick):
        return super().can_recontact(touch, tick) and self.trigger(tick) is not None

    def trigger_record(self, tick):
        return dict(tick=tick, reason=self.trigger(tick), zero_ticks=self.zero_ticks,
                    single_ticks=self.single_ticks, last_new_cell_tick=self.surface.last_observation_tick,
                    measured_cells=len(self.surface.samples))


class ProgressWorker(mesh.MeshWorker):
    def submit(self, data, touch, tick, lead):
        self.observed_trigger = self.servo.trigger_record(tick) if self.servo.can_recontact(touch, tick) else None
        return super().submit(data, touch, tick, lead)

    def install(self, tick, *, initial=False):
        row = super().install(tick, initial=initial)
        if row is not None:
            event = row.get("global_event") or {}
            if "mesh_recontact_query" in event:
                event["mesh_recontact_query"]["trigger"] = self.observed_trigger
                row["mesh_trigger_at_application"] = self.servo.trigger_record(tick)
        return row


def run(folder, obj, seconds=240, device="cuda"):
    reach = policy_reach.PolicyReach()
    if reach.metadata["checkpoint"] != json.loads(policy_reach.PROTOCOL.read_bytes())["checkpoint"]:
        raise ValueError("Reach model and finger policy do not match")
    guide = partial(mesh.PalmGuide, compact=True)
    wrist = partial(ProgressWrist, reach=reach, guide_cls=guide)
    source_modules = {c.__module__ for c in (*ProgressWrist.__mro__, *ProgressWorker.__mro__)
                      if c.__module__.startswith("Module.wrist_active.")}
    sources = {Path(sys.modules[name].__file__) for name in source_modules}
    sources.update(Path(m.__file__) for m in (mesh.palm, mesh.palm_unload, policy_reach, mesh.entry_frontier,
                                              mesh.paired_support, mesh.palm_acquisition, mesh.palm_surface,
                                              mesh.transition_compute))
    return tactile_field.run(folder, obj, seconds, device, wrist_cls=wrist, worker_cls=ProgressWorker,
                             source_files=sorted(sources | {Path(__file__), policy_reach.MODEL}), release=RELEASE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=tuple(mesh.palm.OBJECTS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=240)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    print(run(args.out, args.object, args.seconds, args.device), flush=True)
