"""Supply only wrist commands to the unmodified native controller and acquisition loop."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path as File
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from Module.palmreach6.__main__ import FIXTURES, ROOT as RELEASE
from Module.wrist_active import experiment, sweep_experiment as runtime
from Module.wrist_active.link_clearance import RobotLinkHull
from Module.wrist_active.palm_surface import palm_sites

from .common import DT, FINGERS, HERE, PREVIEW, digest, planning, read, rot
from .planner import Planner
from .trajectory import Path


class Wrist:
    def __init__(self, hand, data, touch, *, models, cfg, variant, **_):
        if hand.objects:
            raise ValueError("Planning hand contains object geometry")
        self.hand, self.cfg = hand, cfg
        self.plate = palm_sites(hand)
        self.hull = RobotLinkHull(hand)
        # Keep extremal vertices of every non-pad collision geometry.
        witnesses = []
        for g in np.unique(self.hull.ids):
            ids = np.flatnonzero(self.hull.ids == g)
            vertices = self.hull.vertices[ids]
            witnesses.extend(ids[np.unique(np.r_[vertices.argmin(0), vertices.argmax(0)])])
        self.witnesses = np.asarray(witnesses)
        origin = data.qpos[:7].copy()
        self.planner = Planner(models, cfg, variant, origin, self.plate[0])
        self.path = Path(0, origin)
        self.clock = self.escape_count = 0
        self.surface, self.route = SimpleNamespace(samples={}), SimpleNamespace(passes=0)

    def command(self, touch):
        self.clock += 1
        return self.path.at(self.clock), self.path.at(self.clock + PREVIEW)

    def phase(self):
        return "scan"


class Worker:
    def __init__(self, wrist, paced):
        self.wrist, self.paced = wrist, paced
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="park3-posterior")
        self.pending = self.target = None
        self.latest = -10000
        self.changed = set()
        self.late = 0
        self.lead = wrist.cfg.lead_ticks
        self.angle_cos = np.cos(np.deg2rad(wrist.cfg.normal_change_deg))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.pool.shutdown(wait=True)
        if self.pending is not None:
            self.pending.result()

    def observe(self, data, touch, tick):
        if not np.isfinite(touch.loads).all() or np.max(touch.loads) >= 8:
            return
        finite = np.isfinite(touch.points).all(1) & np.isfinite(touch.normals).all(1)
        mask = touch.measured_heads & finite & (touch.forces >= .2)
        samples = self.wrist.surface.samples
        for p, n in zip(touch.points[mask], touch.normals[mask]):
            if not .99 < np.linalg.norm(n) < 1.01:
                continue
            key = tuple(np.floor(p/self.wrist.cfg.voxel_m).astype(int))
            if key not in samples or samples[key][1] @ n < self.angle_cos:
                samples[key] = (p.copy(), n.copy())
                self.changed.add(key)

    def submit(self, data, touch, tick, lead):
        w = self.wrist
        actual = data.qpos[:7].copy()
        inv = rot(actual).inv()
        pads = inv.apply(data.geom_xpos[w.hand.pads]-actual[:3])
        links = inv.apply(w.hull.position(data)[w.witnesses]-actual[:3])
        snapshot = np.asarray(list(w.surface.samples.values())).reshape(-1, 2, 3)
        self.due, self.latest = tick+lead, tick
        splice = self.due + (PREVIEW if tick else 0)
        self.pending = self.pool.submit(w.planner.plan, tick, splice, w.path, actual, pads,
                                        np.vstack((w.plate, links)), snapshot[:, 0], snapshot[:, 1], self.target)
        self.changed.clear()

    def install(self, tick, initial=False):
        if self.pending is None or tick < self.due:
            return None
        if self.paced and not initial and not self.pending.done():
            self.late += int(tick == self.due)
            return None
        row, tail = self.pending.result()
        self.pending = None
        row.update(apply_tick=tick, scheduled_tick=self.due)
        if tail is None:
            return row
        if tick > self.due:
            self.lead = min(120, max(self.lead, tick-self.latest+2))
            row["reason"] = "expired_handoff"
            row.pop("selected", None)
            return row
        w = self.wrist
        prefix = w.path.at(np.arange(tick, tail.start))
        w.path = Path(tick, np.vstack((prefix, tail.samples)))
        self.target = {k: row[k] for k in ("point", "normal", "variance")}
        w.route.passes += 1
        return row

    def update(self, data, touch, tick, length):
        self.observe(data, touch, tick)
        row = self.install(tick)
        w = self.wrist
        if self.pending is None and tick-self.latest >= w.cfg.min_replan_ticks and tick+self.lead < length:
            at_end = tick + self.lead + PREVIEW >= w.path.end
            dynamic = w.planner.variant not in ("A", "B", "C")
            if at_end or (dynamic and len(self.changed) >= w.cfg.trigger_cells):
                self.submit(data, touch, tick, self.lead)
        return row


def run(obj, variant, seed, seconds, paced, out, models, cfg):
    if variant not in cfg["experiments"]["variants"] or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Unknown arm or invalid duration")
    if abs(seconds/DT-round(seconds/DT)) > 1e-7:
        raise ValueError("Budget must contain a whole number of control ticks")
    fixture_spec = read(FIXTURES / "fixtures.json")
    case, f = FIXTURES / obj, fixture_spec["objects"][obj]
    for name in ("model.mjb", "initial.npz", "surface.obj"):
        if digest(case/name) != f[name]:
            raise ValueError(f"Paired fixture mismatch: {case/name}")
    if digest(runtime.ROBOT) != fixture_spec["hand_sha256"]:
        raise ValueError("Planning hand differs from release")
    files = [HERE / "park3" / (s+".py") for s in ("belief", "common", "deployment", "geometry", "planner", "probe", "trajectory")]
    names = [s+".npz" for s in FINGERS] + ["gpis.pt"]
    if variant == "FK":
        names.append("fk.npz")
    # Compare original DP, MCC, trajectory scheduler and publisher to the release receipt.
    from .common import ROOT
    original = read(ROOT / "outputs/PARK-NEW/execution-contract.json")["original_files"]
    if any(digest(RELEASE/path) != expected for path, expected in original.items()):
        raise ValueError("Native execution stack differs from its audited release")
    signature = dict(method="PARK-3", variant=variant, config=cfg["planner"],
                     source={p.name: digest(p) for p in files}, models={n: digest(models/n) for n in names},
                     execution=original)
    factory = partial(Wrist, models=models, cfg=planning(cfg), variant=variant)
    seed = f["seed"] if seed is None else seed
    with (patch.object(experiment, "scenario", return_value=(case, case/"surface.obj")),
          patch.object(runtime, "SweepWrist", factory), patch.object(runtime, "WristWorker", Worker),
          patch.object(runtime, "hashes", return_value=signature)):
        result = runtime.run(File(out), obj, f["start"], f"PARK-3-{variant}-dp", seconds, paced, seed)
    p = result["policy"]
    if (result["initial_sha256"] != f["integration_state"] or p["checkpoint_sha256"] != fixture_spec["checkpoint"]
            or p["inference_seed"] != seed or p["action_points"] != 16
            or p["schedule"] != "100ms chunks, 50ms buffered handoff"):
        raise ValueError("Native Finger Policy or initial state mismatch")
    return result
