"""Use the measured contact footprint to place the wrist sweep; output wrist poses only."""

from copy import copy
from types import SimpleNamespace

import numpy as np

from ..acitive_02.active_gpis import VOXEL_M, ObservedSurface
from ..acitive_02.surface_scan import SURFACE_LOAD_N
from .face_wrist import FaceWrist


def contact_origin(touch):
  hits = touch.measured_heads
  if touch.loads.max() >= 8.0 or hits.sum() < 3:
    return np.zeros(3)
  normals = touch.normals[hits]
  normal = normals.mean(axis=0)
  norm = np.linalg.norm(normal)
  if norm < 1.0e-8:
    return np.zeros(3)
  normal = normal / norm
  if (normals @ normal).min() < np.cos(np.deg2rad(35)):
    return np.zeros(3)
  depth = np.median(touch.points[hits] @ normal)
  # The fixture origin is known. A coherent, one-sided footprint defines a second
  # support plane. Use their midpoint as an initial sweep reference.
  return 0.5 * max(0.0, depth) * normal


class FixtureGrid(ObservedSurface):
  """Index new cells on the original fixture lattice while fitting the GP in sweep coordinates."""

  def add(self, touch, tick):
    if touch.loads.max() > SURFACE_LOAD_N:
      return
    for i in np.flatnonzero(touch.measured_heads):
      key = tuple(np.floor(touch.fixture_points[i] / VOXEL_M).astype(int))
      if key not in self.samples:
        self.samples[key] = np.r_[touch.points[i], touch.normals[i]]
        self.last_observation_tick = tick


class ContactCenterWrist(FaceWrist):
  def __init__(self, hand, data, touch, *, center=True, free_depth=True, **options):
    self.origin = contact_origin(touch) if center else np.zeros(3)
    self.translated = bool(self.origin.any())
    self.free_depth = free_depth
    self.empty_ticks = 0
    self.reach_floor = max(
      0.045,
      0.45
      * np.median(
        np.linalg.norm(touch.points[touch.measured_heads] - data.qpos[:3], axis=1)
      ),
    )
    super().__init__(hand, self.local_data(data), self.local_touch(touch), **options)
    if self.translated:
      self.surface = FixtureGrid()

  def local_data(self, data):
    if not self.translated:
      return data
    q = data.qpos.copy()
    q[:3] -= self.origin
    return SimpleNamespace(
      qpos=q, geom_xpos=data.geom_xpos - self.origin, geom_xmat=data.geom_xmat
    )

  def local_touch(self, touch):
    if not self.translated:
      return touch
    local = copy(touch)
    local.fixture_points = touch.points
    local.points = touch.points - self.origin
    return local

  def observe(self, data, touch, tick):
    if self.free_depth:
      free = not touch.measured_heads.any() and touch.loads.max() < 0.35
      self.empty_ticks = self.empty_ticks + 1 if free else 0
      if self.empty_ticks >= 30 and self.radius < self.r_min + 0.008:
        self.r_min = max(self.reach_floor, self.r_min - 0.00045)
    super().observe(self.local_data(data), self.local_touch(touch), tick)

  def plan(self, data, touch, tick):
    row = super().plan(self.local_data(data), self.local_touch(touch), tick)
    row.update(sweep_origin_m=self.origin.tolist(), min_radius_m=float(self.r_min))
    return row

  def command(self, touch):
    now, ahead = super().command(touch)
    if self.translated:
      now[:3] += self.origin
      ahead[:3] += self.origin
    return now, ahead
