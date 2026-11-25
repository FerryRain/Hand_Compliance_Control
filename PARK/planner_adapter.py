"""Deliver frozen PARK endpoints to the PalmReach-6 whole-palm reference interface."""
from contextlib import ExitStack
import json

import numpy as np

from Module.wrist_active import palm_mesh_recontact as mesh
from Module.wrist_active import palm_progress_recontact as progress
from Module.wrist_active.palm_exploration import PalmMove
from Module.wrist_active.palm_landing import Landing
from Module.wrist_active.palm_surface import OUT
from PARK.core.common import rot
from PARK.core.deployment import Worker as GlobalWorker, Wrist as GlobalWrist


def landing(reference, point, normal, site):
    """Represent one world XYZ+wxyz endpoint as a native palm-site landing."""
    attitude = rot(reference)
    center = reference[:3] + attitude.apply(site)
    axis = attitude.apply(OUT)
    # The native trim follows patch + height * axis. Keep the selected wrist
    # transform exact, using the plane through the planner's supported zero crossing.
    height = float((center - point) @ normal / (axis @ normal))
    patch = center - height * axis
    region = PalmMove(center, site, normal)
    return Landing(region, reference[:3], attitude, patch, normal, 1., site, height)


class ReferenceGuide(mesh.PalmGuide):
    def __init__(self, previous, reach):
        super().__init__(previous, reach, compact=True)
        self.reference = None
        self.reference_end = 0

    def apply_reference(self):
        self.palm_goal = self.reference
        self.region = self.reference.region if self.reference is not None else None
        self.holding = self.reference is None
        self.next_tick = self.reference_end

    def choose(self, state, ids, normal, tick, event):
        # The inherited mesh update also calls choose when a recovery query fails.
        # Its normal-path fallback must consume the same published target.
        self.apply_reference()
        event.update(source="park_reference" if self.region else "awaiting_park_reference")

    def update(self, surface, tick, pads, ids, normal, wrist_rotation):
        if len(ids) and self.active_target is None and self.recovery_input is None:
            self.apply_reference()
        return super().update(surface, tick, pads, ids, normal, wrist_rotation)


class PalmWrist(progress.ProgressWrist):
    def __init__(self, hand, data, touch, *, models, cfg, variant, recorder, **options):
        super().__init__(hand, data, touch, reach=None, guide_cls=ReferenceGuide, **options)
        self.global_wrist = GlobalWrist(hand, data, touch, models=models, cfg=cfg, variant=variant)
        self.recorder = recorder
        recorder.nominal.append(data.qpos[:7].copy())

    def command(self, touch):
        result = super().command(touch)
        self.recorder.nominal.append(self.global_wrist.path.at(self.tick + 1).copy())
        return result


class PalmWorker(progress.ProgressWorker):
    def __init__(self, servo, paced):
        super().__init__(servo, paced)
        self.global_worker = GlobalWorker(servo.global_wrist, paced)
        self.reference = None

    def __enter__(self):
        self.stack = ExitStack()
        self.stack.enter_context(self.global_worker)
        super().__enter__()
        return self

    def __exit__(self, *exc):
        try:
            super().__exit__(*exc)
        finally:
            self.stack.close()

    def receive(self, row):
        if row is None:
            return
        if row["reason"] == "new_target":
            endpoint = np.asarray(row["target"])
            last = self.servo.global_wrist.path.at(row["end_tick"])
            np.testing.assert_allclose(endpoint[:3], last[:3], rtol=0, atol=1e-9)
            if (rot(endpoint) * rot(last).inv()).magnitude() > 1e-8:
                raise ValueError("Published path and selected wrist attitude disagree")
            self.reference = landing(endpoint, np.asarray(row["point"]), np.asarray(row["normal"]),
                                     self.servo.selector.guide.site)
        self.servo.recorder.events.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")

    def submit(self, data, touch, tick, lead):
        if tick == 0:
            self.global_worker.observe(data, touch, 0)
            self.global_worker.submit(data, touch, 0, 0)
            self.receive(self.global_worker.install(0, initial=True))
        # There is no outstanding local solve at submission. Give the 20 Hz
        # follower an immutable snapshot while its native worker computes.
        guide = self.servo.selector.guide
        guide.reference = self.reference
        guide.reference_end = self.servo.global_wrist.path.end
        return super().submit(data, touch, tick, lead)

    def update(self, data, touch, tick, length):
        self.receive(self.global_worker.update(data, touch, tick, length))
        return super().update(data, touch, tick, length)
