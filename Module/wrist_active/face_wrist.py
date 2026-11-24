"""Align a moving pad footprint to broad faces identified from physical touches."""

from itertools import combinations

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.aude import unit
from ..acitive_02.surface_scan import PadSupport, align, limit
from ..i04_geometric_traversal.physics import quaternion, rotation
from .load_guard import GuardedWrist
from .sweep import DT, LOAD_LIMIT, PLAN_TICKS, PREVIEW_S, Spiral
from .wrist_worker import WristWorker


def broad_face(touch):
  if touch.loads.max() >= LOAD_LIMIT:
    return None
  for triplet in combinations(np.flatnonzero(touch.measured_heads), 3):
    ids = np.array(triplet)
    normals, points = touch.normals[ids], touch.points[ids]
    if (normals @ normals.T).min() < 0.998:
      continue
    normal = unit(normals.mean(axis=0))
    span = np.linalg.norm(points[:, None] - points, axis=2).max()
    if span >= 0.055 and np.ptp(points @ normal) <= 0.0012:
      return ids, normal
  return None


class TactileEnvelope:
  """Tangent supports of the live tactile map; the known fixture origin is inside."""

  def fit(self, surface):
    points, normals = surface.points, surface.normals
    # Bound the number of directions while keeping opposite faces separate.
    _, indices = np.unique(np.rint(normals / 0.16).astype(int), axis=0, return_index=True)
    self.normals = normals[indices]
    self.supports = np.max(points @ self.normals.T, axis=0)
    self.outer = 1.18 * np.max(np.linalg.norm(points, axis=1))

  def predict(self, ray):
    dot = self.normals @ ray
    # A reflected support is only an initial depth proposal for an unseen side.
    # Replace it as soon as any outward-facing local measurements support it.
    bounds = np.where(dot > 0.25, self.supports / np.maximum(dot, 0.25), np.inf)
    back = np.where(dot < -0.4, (self.supports + 0.012) / np.maximum(-dot, 0.4), np.inf)
    best = int(np.argmin(bounds))
    other = int(np.argmin(back))
    radius = min(bounds[best], back[other], self.outer)
    weights = np.maximum(0.0, 1.0 - (bounds - radius) / 0.020) ** 2
    if weights.sum() > 0.02:
      normal = unit(weights @ self.normals)
    elif back[other] <= self.outer:
      normal = -self.normals[other]
    else:
      normal = ray
    return radius * ray, normal


class FaceStrips:
  """Continue a whole row of pads along poorly sampled lanes on measured faces."""

  def __init__(self, width):
    self.width = width
    self.target = None
    self.ray = None
    self.tried = ()
    self.traversing = False

  def choose(self, wrist):
    surface = wrist.surface
    normals, points = surface.normals, surface.points
    _, first, counts = np.unique(
      np.rint(normals / 0.20).astype(int), axis=0, return_index=True, return_counts=True
    )
    planes = []
    best, chosen = -np.inf, None
    for idx in first[np.argsort(-counts)]:
      normal = normals[idx]
      if any(normal @ n > 0.93 for n in planes):
        continue
      group = normals @ normal > 0.965
      p = points[group]
      if len(p) < 18:
        continue
      height = np.median(p @ normal)
      p = p[abs(p @ normal - height) < 0.004]
      if len(p) < 16:
        continue
      row = wrist.face_axis if abs(normal @ wrist.face_axis) < 0.72 else wrist.first_row
      row = unit(row - (row @ normal) * normal)
      along = unit(np.cross(normal, row))
      v0, v1 = np.percentile(p @ along, [2, 98])
      u0, u1 = np.percentile(p @ row, [2, 98])
      if min(v1 - v0, u1 - u0) < 0.035:
        continue
      planes.append(normal)
      lanes = np.linspace(u0, u1, int(np.ceil((u1 - u0) / 0.012)) + 1)
      travel = np.linspace(v0, v1, 12)
      lines = normal * height + lanes[:, None, None] * row + travel[None, :, None] * along
      pad_widths = np.array([-0.45, 0.0, 0.45]) * self.width
      probes = lines[:, :, None, :] + pad_widths[None, None, :, None] * row
      feasible = (lanes[:, None] + pad_widths >= u0 - 0.005) & (lanes[:, None] + pad_widths <= u1 + 0.005)
      gaps = np.minimum(surface.tree.query(probes)[0], 0.028)
      score = (gaps * feasible[:, None, :]).mean(axis=(1, 2))
      endpoints = lines[:, [0, -1]]
      travel_cost = np.linalg.norm(endpoints - wrist.goal_point, axis=2)
      score *= np.minimum(1.0, (v1 - v0) / 0.11) / (1.0 + 2.0 * travel_cost.min(axis=1))
      centers = lines.mean(axis=1)
      if self.tried:
        closest = np.linalg.norm(centers[:, None, :] - np.asarray(self.tried), axis=2).min(axis=1)
        score *= np.clip(closest / 0.022, 0.12, 1.0)
      winner = int(score.argmax())
      if score[winner] > best:
        ends = endpoints[winner]
        if travel_cost[winner, 1] < travel_cost[winner, 0]:
          ends = ends[::-1]
        best = score[winner]
        chosen = (ends[0].copy(), ends[1].copy(), normal.copy())
      if len(planes) >= 10:
        break
    return chosen

  def advance(self, wrist, tick):
    if self.ray is None:
      self.ray = unit(wrist.goal_point)
    if self.target is None or tick >= self.due:
      self.target = self.choose(wrist)
      if self.target is None:
        return None
      start, end, _ = self.target
      self.tried = (*self.tried[-18:], 0.5 * (start + end))
      self.traversing = False
      self.progress = 0.0
      self.due = tick + round(
        100
        * (
          2.0
          + np.arccos(np.clip(self.ray @ unit(start), -1.0, 1.0)) / 0.59
          + np.linalg.norm(end - start) / 0.060
        )
      )
    start, end, normal = self.target
    if not self.traversing:
      turn = align(self.ray, unit(start)).as_rotvec()
      self.ray = Rotation.from_rotvec(limit(turn, 0.60 * PLAN_TICKS * DT)).apply(self.ray)
      if np.linalg.norm(turn) > 0.095:
        point, n = wrist.envelope.predict(self.ray)
        return point, n
      self.traversing = True
    length = np.linalg.norm(end - start)
    self.progress = min(length, self.progress + (0.072 if not wrist.braking else 0.012) * PLAN_TICKS * DT)
    point = start + (self.progress / length) * (end - start)
    self.ray = unit(point)
    if self.progress >= length:
      self.target = None
    return point, normal


class FaceWrist(GuardedWrist):
  """Use the existing guarded sweep until a measured multi-pad plane is present."""

  def __init__(self, hand, data, touch, **options):
    super().__init__(hand, data, touch, **options)
    self.face_ids = None
    self.face_tick = None
    self.envelope = TactileEnvelope()
    self.envelope.offset = 0.0
    self.goal_point = data.qpos[:3].copy()
    self.goal_attitude = self.attitude
    self.spin = np.zeros(3)
    self.goal_normal = self.radial.copy()
    self.position_ref = self.goal_point.copy()
    self.pivot = np.zeros(3)
    self.local_heads = np.zeros((4, 3))
    self.contact_tilt = np.zeros(3)
    self.face_ready = False
    self.vertices = PadSupport(hand).vertices
    self.normal_ticks = np.full(4, -10000)
    self.contact_normals = np.zeros((4, 3))
    self.unload_direction = self.radial.copy()
    self.examples = ()
    self.quorum = 1.0

  def observe(self, data, touch, tick):
    super().observe(data, touch, tick)
    if self.face_ids is not None:
      hits = touch.measured_heads
      light = touch.loads.max() < LOAD_LIMIT
      self.quorum += 0.025 * (float(light and hits.sum() >= 2) - self.quorum)
      if tick % 20 == 0 and light and hits.sum() >= 3:
        selected = np.flatnonzero(hits)
        normals = touch.normals[selected]
        group = normals @ normals.T > 0.90
        keep = group[np.argmax(group.sum(axis=1))]
        if keep.sum() >= 2:
          normal = unit(normals[keep].mean(axis=0))
          local = rotation(data.qpos[3:7]).inv().apply(normal)
          sample = (data.qpos[7:].copy(), local, abs(normal @ self.face_axis))
          self.examples = (*self.examples[-63:], sample)
      self.contact_normals[hits] = touch.normals[hits]
      self.normal_ticks[hits] = tick
      region = int(touch.loads.argmax())
      if region < 4:
        radial = unit(data.geom_xpos[self.hand.pads[region]])
        measured = self.contact_normals[region]
        if tick - self.normal_ticks[region] <= 60 and radial @ measured > 0.08:
          self.unload_direction = unit(0.88 * measured + 0.12 * radial)
        else:
          self.unload_direction = radial
      else:
        self.unload_direction = unit(data.qpos[:3])
      self.escape_ticks = 0
      self.last_escape = tick
      self.escape_count = self.brake_events
      return
    if tick % PLAN_TICKS or not self.gpis or not self.recontact:
      return
    patch = broad_face(touch)
    if patch is None:
      return
    self.face_ids, normal = patch
    self.face_tick = tick
    live = rotation(data.qpos[3:7])
    self.local_normal = live.inv().apply(normal)
    pts = touch.points[self.face_ids]
    self.face_axis = normal.copy()
    self.first_row = unit(pts[-1] - pts[0])
    self.strips = FaceStrips(float(np.ptp(pts @ self.first_row)))
    self.local_row = live.inv().apply(self.first_row)
    self.route = Spiral(normal, pts[-1] - pts[0])
    self.route.pitch = 0.42
    self.goal_normal = normal.copy()
    self.contact_normals[touch.measured_heads] = touch.normals[touch.measured_heads]
    self.normal_ticks[touch.measured_heads] = tick
    self.goal_attitude = self.attitude
    self.position_ref = self.radius * self.radial + self.shift
    self.pivot = live.inv().apply(pts.mean(axis=0) - data.qpos[:3])
    self.goal_point = self.position_ref + live.apply(self.pivot)

  def phase(self):
    if not self.face_ready:
      return super().phase()
    if self.braking:
      return "withdraw"
    return "approach" if self.bad_ticks >= 18 else "scan"

  def plan(self, data, touch, tick):
    if self.face_ids is None:
      return super().plan(data, touch, tick)
    if self.surface.tree is None or tick - getattr(self.surface, "fit_tick", -1000) >= 80:
      self.surface.refresh()
      self.surface.fit_tick = tick
      self.envelope.fit(self.surface)
    step = PLAN_TICKS * DT
    rate = 0.49 if not self.braking else 0.12
    lane = self.strips.advance(self, tick) if self.route.passes >= 2 else None
    if lane is None:
      ray = self.route.advance(rate * step, self.surface.points)
      anchor, normal = self.envelope.predict(ray)
    else:
      anchor, normal = lane
      ray = unit(anchor)
    patch = self.surface.patch(anchor, preferred_normal=normal)
    if patch is not None:
      value, gradient = patch.predict(anchor)
      local_normal = unit(gradient[0])
      weight = np.clip(1.0 - np.linalg.norm(anchor - patch.anchor) / 0.040, 0.0, 1.0)
      projection = np.clip(value[0] / max(local_normal @ ray, 0.35), -0.009, 0.009)
      anchor = anchor - weight * projection * ray
      normal = unit((1.0 - weight) * normal + weight * local_normal)
    live = rotation(data.qpos[3:7])
    heads = []
    current_normal = live.apply(self.local_normal)
    for geom in self.hand.pads:
      vertices = self.vertices[int(geom)] @ data.geom_xmat[geom].reshape(3, 3).T
      heights = vertices @ current_normal
      heads.append(data.geom_xpos[geom] + vertices[heights <= heights.min() + 0.0004].mean(axis=0))
    local = live.inv().apply(np.asarray(heads) - data.qpos[:3])
    # Level the row of measured pads about its center; the short transverse
    # dimension stays close to the orientation that produced the flat contact.
    centered = local - local.mean(axis=0)
    height = centered @ self.local_normal
    slope = np.cross(centered, self.local_normal)
    tilt = np.linalg.solve(slope.T @ slope + 0.025**2 * np.eye(3), -slope.T @ height)
    local_normal = Rotation.from_rotvec(-limit(tilt, 0.65)).apply(self.local_normal)
    if self.examples:
      joints, directions, slopes = (np.asarray(values) for values in zip(*self.examples, strict=True))
      error = np.mean((joints - data.qpos[7:]) ** 2, axis=1) / 0.15**2
      error += ((slopes - abs(normal @ self.face_axis)) / 0.33) ** 2
      weight = np.exp(-0.5 * error) * np.maximum(0.0, directions @ local_normal) ** 3
      gain = 0.30 * (1.0 - self.quorum) * min(1.0, weight.sum() / 2.0)
      if gain > 0.001:
        local_normal = unit((1.0 - gain) * local_normal + gain * unit(weight @ directions))
    base = self.attitude * Rotation.from_rotvec(-self.contact_tilt)
    adjust = align(base.apply(local_normal), normal).as_rotvec()
    desired = Rotation.from_rotvec(limit(adjust, 1.35 * step)) * base
    # Spread the row of fingers across adjacent scan lines on a side wall.
    direction = self.face_axis if abs(normal @ self.face_axis) < 0.72 else self.first_row
    across = unit(direction - (direction @ normal) * normal)
    row = desired.apply(self.local_row)
    row = unit(row - (row @ normal) * normal)
    across = across if across @ row >= 0 else -across
    yaw = np.arctan2(np.cross(row, across) @ normal, row @ across)
    desired = Rotation.from_rotvec(np.clip(yaw, -0.12, 0.12) * normal) * desired
    self.goal_attitude = desired
    self.goal_normal = normal
    self.pivot = local[self.face_ids].mean(axis=0)
    self.local_heads = local
    path = self.route.at(self.route.angle + 0.002) - self.route.at(self.route.angle - 0.002)
    across = unit(np.cross(normal, path))
    offsets = np.array([0.0, -0.014, 0.014, -0.025, 0.025])
    points = anchor + desired.apply(local - self.pivot)
    queries = points[None, :, :] + offsets[:, None, None] * across
    distance = self.surface.tree.query(queries)[0]
    novelty = np.minimum(distance, 0.018)
    if patch is not None:
      variance = patch.predict(queries.reshape(-1, 3), variance=True).reshape(distance.shape)
      novelty *= 0.65 + 0.35 * np.sqrt(variance)
    scores = novelty.mean(axis=1) - 0.12 * abs(offsets) - 0.09 * abs(offsets - self.envelope.offset)
    offset = offsets[scores.argmax()]
    self.envelope.offset += np.clip(offset - self.envelope.offset, -0.007, 0.007)
    self.goal_point = anchor if lane is not None else anchor + self.envelope.offset * across
    self.face_ready = True
    return dict(
      tick=tick,
      pass_index=self.route.passes,
      arc=self.route.angle,
      cells=len(self.surface.samples),
      last_new_tick=self.surface.last_observation_tick,
      escape_count=self.escape_count,
      choice=0,
      supported=int(touch.measured_heads.sum()),
      face_tick=self.face_tick,
      grasp_examples=len(self.examples),
      face_strip=lane is not None,
      load_brake=self.braking,
      brake_events=self.brake_events,
    )

  def command(self, touch):
    if not self.face_ready:
      return super().command(touch)
    peak = float(touch.loads.max())
    hits = int(touch.measured_heads.sum())
    normal = self.goal_normal
    if touch.measured_heads.any():
      measured = unit(touch.normals[touch.measured_heads].mean(axis=0))
      if measured @ normal > 0.2:
        normal = unit(0.64 * normal + 0.36 * measured)
    if self.braking:
      step = min(0.16, 0.020 + 0.006 * peak) * DT if peak >= 5.0 else 0.0
      relief = self.unload_direction if normal @ self.unload_direction < 0.55 else normal
      target = self.position_ref + step * relief
      self.position_ref = unit(target) * np.clip(np.linalg.norm(target), self.r_min, self.r_max)
      current = np.r_[self.position_ref, quaternion(self.attitude)]
      return current, current.copy()
    center = self.position_ref + self.attitude.apply(self.pivot)
    error = self.goal_point - center
    if peak > 4.0:
      speed = min(0.12, 0.013 * (peak - 3.0))
    elif hits < 2:
      speed = -0.060 if peak < 1.3 else -0.014
    else:
      speed = 0.006 * (peak - 1.6)
    if hits == 0 and peak < 1.0:
      speed = np.clip(0.9 * (error @ normal) - 0.045, -0.105, 0.04)
    tangent = error - (error @ normal) * normal
    tangent = limit(3.2 * tangent, 0.12) * np.clip((12.0 - peak) / 7.0, 0.08, 1.0)
    # A loaded pad rotates away from the surface about the footprint, lowering
    # the opposite pads. The integral also works when new cells keep arriving.
    loads = np.minimum(touch.loads[:4], 7.0)
    arms = self.local_heads - self.local_heads.mean(axis=0)
    moment = np.cross(arms, self.attitude.inv().apply(normal)).T @ loads
    self.contact_tilt = limit((1.0 - 0.55 * DT) * self.contact_tilt + DT * limit(4.0 * moment, 0.85), 0.50)
    adjusted = self.goal_attitude * Rotation.from_rotvec(self.contact_tilt)
    turn = (adjusted * self.attitude.inv()).as_rotvec()
    angular = limit(5.0 * turn + self.attitude.apply(3.2 * moment), 1.25)
    updated = Rotation.from_rotvec(angular * DT) * self.attitude
    target = center - updated.apply(self.pivot) + DT * (tangent + speed * normal)
    target = unit(target) * np.clip(np.linalg.norm(target), self.r_min, self.r_max)
    self.position_ref = self.position_ref + limit(target - self.position_ref, 0.002)
    self.attitude = updated
    current = np.r_[self.position_ref, quaternion(self.attitude)]
    future = Rotation.from_rotvec(angular * PREVIEW_S) * updated
    preview = np.r_[
      self.position_ref
      + updated.apply(self.pivot)
      - future.apply(self.pivot)
      + PREVIEW_S * tangent
      + 0.04 * speed * normal,
      quaternion(future),
    ]
    return current, preview


class FaceWorker(WristWorker):
  command_fields = (
    *WristWorker.command_fields,
    "goal_point",
    "goal_normal",
    "goal_attitude",
    "pivot",
    "local_heads",
    "face_ready",
  )
