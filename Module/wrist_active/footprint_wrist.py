"""Initialize wrist height and orientation from a coherent measured fingertip footprint."""

import numpy as np

from ..acitive_02.aude import unit
from ..i04_geometric_traversal.physics import rotation
from .contact_center import ContactCenterWrist
from .face_wrist import FaceStrips
from .sweep import Spiral


class FootprintWrist(ContactCenterWrist):
  def __init__(self, hand, data, touch, **options):
    super().__init__(hand, data, touch, **options)
    ids = np.flatnonzero(touch.measured_heads)
    if len(ids) != 4 or touch.loads.max() >= 8:
      return
    normals = touch.normals[ids]
    normal = unit(normals.mean(axis=0))
    points = self.local_touch(touch).points[ids]
    row = points[-1] - points[0]
    if (
      min(normals @ normal) < np.cos(np.deg2rad(32))
      or np.ptp(points @ normal) > 0.020
      or np.linalg.norm(row) < 0.055
    ):
      return
    live = rotation(data.qpos[3:7])
    self.face_ids = ids
    self.face_tick = 0
    self.local_normal = live.inv().apply(normal)
    self.face_axis = normal.copy()
    self.first_row = unit(row)
    self.strips = FaceStrips(float(np.ptp(points @ self.first_row)))
    self.local_row = live.inv().apply(self.first_row)
    self.route = Spiral(normal, row)
    self.route.pitch = 0.42
    self.goal_normal = normal.copy()
    self.contact_normals[ids] = normals
    self.normal_ticks[ids] = 0
    self.goal_attitude = self.attitude
    self.position_ref = self.radius * self.radial + self.shift
    self.pivot = live.inv().apply(points.mean(axis=0) - self.position_ref)
    self.goal_point = self.position_ref + live.apply(self.pivot)
