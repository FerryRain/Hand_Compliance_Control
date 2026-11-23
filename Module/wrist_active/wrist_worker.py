"""Compute the next wrist segment off the servo thread from a causal snapshot."""
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from itertools import islice
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.active_gpis import ObservedSurface
from .sweep import DT, PLAN_TICKS


class WristWorker:
    lead = 5
    command_fields = ('base_attitude','local_tilt','feed_speed','twist','spin','shift_velocity')

    def __init__(self, servo, paced):
        self.servo = servo
        self.map = ObservedSurface()
        self.delta = {}
        self.paced = paced
        self.pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='wrist-gpis')
        self.pending = None
        self.due = 0
        self.late = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pool.shutdown(wait=True)
        if self.pending is not None:
            self.pending.result()

    def observe(self, data, touch, tick):
        samples = self.servo.surface.samples
        before = len(samples)
        self.servo.observe(data,touch,tick)
        # Up to four new cells per packet. Keep the original insertion order.
        keys = list(islice(reversed(samples),len(samples)-before))
        self.delta.update((key,samples[key]) for key in reversed(keys))

    def submit(self, data, touch, tick, lead):
        snapshot = copy(self.servo)
        snapshot.surface = self.map
        snapshot.offsets = self.servo.offsets.copy()
        snapshot.last_normals = self.servo.last_normals.copy()
        if lead:
            duration = lead*DT
            snapshot.shift = self.servo.shift+self.servo.shift_velocity*duration
            snapshot.radial = Rotation.from_rotvec(snapshot.twist*duration).apply(self.servo.radial)
            snapshot.attitude = Rotation.from_rotvec(self.servo.spin*duration)*self.servo.attitude
            snapshot.radius = np.clip(self.servo.radius+self.servo.feed_speed*duration,
                                      self.servo.r_min,self.servo.r_max)
        else:
            snapshot.shift = self.servo.shift.copy()
        sensed = SimpleNamespace(qpos=data.qpos.copy(),geom_xpos=data.geom_xpos.copy(),
                                 geom_xmat=data.geom_xmat.copy())
        delta, last = self.delta,self.servo.surface.last_observation_tick
        self.delta = {}
        def compute():
            began = perf_counter()
            self.map.samples.update(delta)
            self.map.last_observation_tick = last
            row = snapshot.plan(sensed,touch,tick)
            row['planning_ms'] = 1000*(perf_counter()-began)
            return row,[getattr(snapshot,field) for field in self.command_fields]
        self.due = tick+lead
        self.pending = self.pool.submit(compute)

    def install(self, tick, *, initial=False):
        if self.pending is None or tick<self.due:
            return None
        if self.paced and not initial and not self.pending.done():
            self.late += int(tick==self.due)
            # Hold the previous endpoint while load feedback continues.
            self.servo.twist = self.servo.spin = self.servo.shift_velocity = np.zeros(3)
            self.servo.feed_speed = 0.
            return None
        row, fields = self.pending.result()
        self.pending = None
        for name,value in zip(self.command_fields,fields):
            setattr(self.servo,name,value)
        row.update(apply_tick=tick,scheduled_tick=self.due)
        return row

    def update(self, data, touch, tick, length):
        self.observe(data,touch,tick)
        row = self.install(tick)
        if tick%PLAN_TICKS==PLAN_TICKS-self.lead and tick+self.lead<length and self.pending is None:
            self.submit(data,touch,tick,self.lead)
        return row
