"""Plan wrist poses from the live fingertip map and measured wrist loads."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..acitive_02.aude import unit
from ..acitive_02.surface_scan import align, limit
from ..i04_geometric_traversal.physics import quaternion, rotation
from .face_wrist import FaceWorker
from .footprint_wrist import FootprintWrist
from .sweep import DT, PLAN_TICKS, Spiral


def cylindrical_footprint(points, normals):
  """Recover the straight direction only when four measured tangents support it."""
  _, values, basis = np.linalg.svd(normals, full_matrices=False)
  if values[1] < 0.45 or values[2] > 0.055:
    return None
  axis = basis[-1]
  flat = points - np.outer(points @ axis, axis)
  radial = normals - np.outer(normals @ axis, axis)
  radial /= np.linalg.norm(radial, axis=1)[:, None]
  centered = radial - radial.mean(axis=0)
  radius = np.sum((flat - flat.mean(axis=0)) * centered) / np.sum(centered**2)
  centers = flat - radius * radial
  residual = np.sqrt(np.mean(np.sum((centers - centers.mean(axis=0)) ** 2, axis=1)))
  if 0.035 < radius < 0.28 and residual < 0.005 and np.ptp(points @ axis) > 0.045:
    return axis, unit(radial.mean(axis=0))
  return None


class ContactFrontiers:
  """Select an under-observed direction and follow the estimated surface to it."""

  def __init__(self):
    i = np.arange(640) + 0.5
    z = 1 - 2 * i / len(i)
    a = i * np.pi * (3 - np.sqrt(5))
    self.rays = np.column_stack(
      (np.sqrt(1 - z * z) * np.cos(a), np.sqrt(1 - z * z) * np.sin(a), z)
    )
    self.ray = None
    self.target = None
    self.visited = ()
    self.due = 0
    self.arrived = None
    self.lanes = None
    self.tested_shape = False
    self.measured_axis = None

  def local_surface(self, rays, anchors):
    distances, indices = self.direction_tree.query(rays, k=6)
    normals, points = self.normals[indices], self.points[indices]
    radii = self.depth[indices]
    cosine = (normals * rays[..., None, :]).sum(axis=-1)
    plane = np.sum(normals * points, axis=-1) / np.maximum(cosine, 0.35)
    projected = np.where(
      cosine > 0.35, np.clip(plane, radii - 0.025, radii + 0.025), radii
    )
    w = 1.0 / (distances + 0.020) ** 2
    w /= w.sum(axis=-1, keepdims=True)
    trust = np.clip((0.32 - distances[..., 0]) / 0.19, 0.0, 1.0)
    observed = rays * (w * projected).sum(axis=-1)[..., None]
    return (
      anchors + trust[..., None] * (observed - anchors),
      (w[..., None] * normals).sum(axis=-2),
      trust,
    )

  def choose(self, wrist, tick):
    env, surface = wrist.envelope, wrist.surface
    dot = self.rays @ env.normals.T
    radius = np.minimum(
      np.where(dot > 0.25, env.supports / np.maximum(dot, 0.25), np.inf).min(axis=1),
      np.where(dot < -0.4, (env.supports + 0.012) / np.maximum(-dot, 0.4), np.inf).min(
        axis=1
      ),
    ).clip(0.018, max(env.outer, 0.018))
    points = radius[:, None] * self.rays
    self.points, self.normals = surface.points, surface.normals
    self.depth = np.maximum(np.linalg.norm(self.points, axis=1), 0.001)
    self.direction_tree = cKDTree(self.points / self.depth[:, None])
    points, _, _ = self.local_surface(self.rays, points)
    # A footprint-sized neighborhood scores a lane, so a tiny isolated hole
    # does not outweigh a whole unsampled side of the surface.
    gap = np.minimum(surface.tree.query(points)[0], 0.075)
    neighbors = self.rays @ self.rays.T > 0.94
    worth = 0.55 * gap + 0.45 * ((neighbors @ gap) / neighbors.sum(axis=1))
    angle = np.arccos(np.clip(self.rays @ self.ray, -1, 1))
    worth /= 1 + 0.24 * angle
    if self.visited:
      directions, stamps = zip(*self.visited, strict=True)
      age = (tick - np.array(stamps)) * DT
      memory = np.maximum(0, (self.rays @ np.array(directions).T - 0.65) / 0.35)
      worth *= 1 - 0.93 * np.max(memory * np.exp(-age / 115), axis=1)
    winner = int(worth.argmax())
    self.target = self.rays[winner]
    self.visited = (*self.visited[-38:], (self.target, tick))
    self.due = tick + round((angle[winner] / 0.43 + 2.4) / DT)
    self.arrived = None

  def advance(self, wrist, tick):
    if not self.tested_shape and len(wrist.surface.points) >= 250:
      self.tested_shape = True
      self.lanes = AxialLanes.from_contacts(
        wrist.surface, wrist.goal_point, self.measured_axis
      )
    if self.lanes is not None:
      return self.lanes.advance(wrist)
    if self.ray is None:
      self.ray = unit(wrist.goal_point)
    if (
      self.target is None
      or tick >= self.due
      or (self.arrived is not None and tick - self.arrived >= 95)
    ):
      self.choose(wrist, tick)
    turn = align(self.ray, self.target).as_rotvec()
    if np.linalg.norm(turn) < 0.07 and self.arrived is None:
      self.arrived = tick
    self.ray = Rotation.from_rotvec(limit(turn, 0.46 * PLAN_TICKS * DT)).apply(self.ray)
    anchor, normal = wrist.envelope.predict(self.ray)
    point, measured, trust = self.local_surface(self.ray, anchor)
    return point, unit((1 - 0.8 * trust) * normal + 0.8 * trust * measured)


class AxialLanes:
  """Run end-to-end tracks around the long axis when the measured cloud supports one."""

  @classmethod
  def from_contacts(cls, surface, initial, measured_axis):
    points = surface.points
    _, _, frame = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    span = np.ptp(np.percentile(points @ frame.T, [1, 99], axis=0), axis=0)
    if span[0] < 1.65 * np.mean(span[1:]):
      return None
    if measured_axis is not None:
      side = frame[np.argmin(np.abs(frame @ measured_axis))]
      side = unit(side - (side @ measured_axis) * measured_axis)
      frame = np.array([measured_axis, side, np.cross(measured_axis, side)])
    return cls(surface, frame, initial)

  def __init__(self, surface, frame, initial):
    self.frame = frame
    self.update_profile(surface)
    local = initial @ frame.T
    self.z = float(np.clip(local[0], self.low, self.high))
    offset = local[1:] - self.center(self.z)
    self.angle = np.arctan2(offset[1], offset[0])
    self.travel = -1 if self.z < (self.low + self.high) * 0.5 else 1
    self.crossing = False
    self.history = []
    self.angles = np.arange(32) * (2 * np.pi / 32)

  def center(self, z):
    return np.stack([np.interp(z, self.stations, v) for v in self.centers.T], axis=-1)

  def update_profile(self, surface):
    self.fit_tick = surface.fit_tick
    self.points = points = surface.points @ self.frame.T
    self.normals = surface.normals @ self.frame.T
    lo, hi = np.percentile(points[:, 0], [0.3, 99.7])
    self.low, self.high = lo + 0.004, hi - 0.004
    self.stations = np.linspace(self.low, self.high, 18)
    centers = []
    for z in self.stations:
      group = np.abs(points[:, 0] - z) < max(0.018, (hi - lo) / 15)
      if group.sum() < 15:
        closest = np.argsort(np.abs(points[:, 0] - z))[:32]
        side = points[closest, 1:]
      else:
        side = points[group, 1:]
      centers.append(np.percentile(side, [2, 98], axis=0).mean(axis=0))
    self.centers = np.array(centers)
    xy = points[:, 1:] - self.center(points[:, 0])
    self.radii = np.maximum(np.linalg.norm(xy, axis=1), 0.001)
    directions = xy / self.radii[:, None]
    self.scale = np.median(self.radii)
    self.tree = cKDTree(np.column_stack((points[:, 0], self.scale * directions)))

  def side(self, z, angle):
    z, angle = np.broadcast_arrays(z, angle)
    direction = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    center = self.center(z)
    features = np.concatenate((z[..., None], self.scale * direction), axis=-1)
    dist, idx = self.tree.query(features, k=8)
    normals = self.normals[idx]
    cosine = (normals[..., 1:] * direction[..., None, :]).sum(axis=-1)
    base = np.concatenate((z[..., None], center), axis=-1)
    projection = ((self.points[idx] - base[..., None, :]) * normals).sum(axis=-1)
    depth = np.where(
      cosine > 0.35,
      np.clip(
        projection / np.maximum(cosine, 0.35),
        self.radii[idx] - 0.028,
        self.radii[idx] + 0.028,
      ),
      self.radii[idx],
    )
    weight = (0.12 + np.maximum(cosine, 0)) / (dist + 0.007) ** 2
    weight /= weight.sum(axis=-1, keepdims=True)
    point = base.copy()
    point[..., 1:] += (
      np.maximum(0.012, (weight * depth).sum(axis=-1))[..., None] * direction
    )
    normal = (normals * weight[..., None]).sum(axis=-2)
    normal[..., 1:] += 0.22 * direction
    normal /= np.maximum(1e-7, np.linalg.norm(normal, axis=-1, keepdims=True))
    return point @ self.frame, normal @ self.frame

  def next_angle(self, surface):
    stations = np.linspace(self.low, self.high, 22)
    points, _ = self.side(stations[:, None], self.angles)
    gaps = np.minimum(surface.tree.query(points)[0], 0.045)
    worth = gaps.mean(axis=0) + 0.35 * np.percentile(gaps, 80, axis=0)
    # Smooth travel through the end cap also covers it between axial tracks.
    if self.history:
      proximity = 0.5 + 0.5 * np.cos(self.angles[:, None] - np.array(self.history))
      worth *= 1 - 0.90 * (
        proximity**8 * np.exp(-np.arange(len(self.history))[::-1] / 5)
      ).max(axis=1)
    chosen = float(self.angles[np.argmax(worth)])
    self.history = [*self.history[-14:], chosen]
    return chosen

  def advance(self, wrist):
    if wrist.surface.fit_tick - self.fit_tick >= 240:
      self.update_profile(wrist.surface)
    dt = PLAN_TICKS * DT
    if self.crossing:
      self.fraction = min(1.0, self.fraction + dt * 0.060 / self.chord)
      u = self.fraction
      left, n0 = self.side(self.z, self.angle)
      right, n1 = self.side(self.z, self.next_lane)
      cap = self.travel * self.frame[0]
      mid = np.r_[self.z + self.travel * 0.007, self.center(self.z)] @ self.frame
      point = (1 - u) ** 2 * left + 2 * u * (1 - u) * mid + u * u * right
      strength = np.sin(np.pi * u)
      normal = unit((1 - strength) * ((1 - u) * n0 + u * n1) + strength * cap)
      if u >= 1:
        self.angle, self.travel, self.crossing = self.next_lane, -self.travel, False
      return point, normal
    end = self.high if self.travel == 1 else self.low
    speed = 0.050 if wrist.empty >= 20 else 0.071
    self.z += np.clip(end - self.z, -speed * dt, speed * dt)
    point, normal = self.side(self.z, self.angle)
    if abs(end - self.z) < 0.001:
      self.next_lane = self.next_angle(wrist.surface)
      next_point, _ = self.side(self.z, self.next_lane)
      self.chord = max(0.095, np.linalg.norm(next_point - point))
      self.crossing, self.fraction = True, 0.0
    return point, normal


class FrontierWrist(FootprintWrist):
  def __init__(self, hand, data, touch, **options):
    super().__init__(hand, data, touch, **options)
    self.frontiers = ContactFrontiers()
    self.tick = 0
    self.saved_pose = None
    self.saved_tick = 0
    self.empty = 0
    self.last_rescue = -500
    self.rescue_until = 0
    self.clearance_ticks = 0
    self.clear_frames = 0
    self.relief_count = 0
    self.recontact_count = 0
    self.force_before = 0.0
    self.last_palm_contact = None
    self.outward = self.radial.copy()
    self.recent_motion = np.zeros(3)
    self.actual_position = data.qpos[:3].copy()
    self.last_command = self.actual_position.copy()
    self.last_rotation = self.attitude
    self.palm_planes = None
    geom = hand.palm_geom
    model = hand.model
    mesh = model.geom_dataid[geom]
    begin = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[begin : begin + model.mesh_vertnum[mesh]]
    directions = np.array(
      [[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
    )
    vertices = vertices[np.unique(np.argmax(vertices @ directions.T, axis=0))]
    world = data.geom_xpos[geom] + vertices @ data.geom_xmat[geom].reshape(3, 3).T
    self.palm_vertices = self.attitude.inv().apply(world - self.actual_position)
    ids = np.flatnonzero(touch.measured_heads)
    if self.face_ids is None and len(ids) == 4 and touch.loads.max() < 8:
      points = self.local_touch(touch).points[ids]
      fit = cylindrical_footprint(points, touch.normals[ids])
      if fit is not None:
        row, normal = fit
        self.frontiers.measured_axis = row
        self.face_ids, self.face_tick = ids, 0
        self.face_axis, self.first_row = normal, row
        self.local_normal = self.attitude.inv().apply(normal)
        self.local_row = self.attitude.inv().apply(row)
        self.route = Spiral(normal, row)
        self.route.pitch = 0.42
        self.goal_normal = normal.copy()
        self.normal_ticks[ids] = 0
        self.contact_normals[ids] = touch.normals[ids]
        self.position_ref = self.radius * self.radial + self.shift
        self.pivot = self.attitude.inv().apply(points.mean(axis=0) - self.position_ref)
        self.goal_point = self.position_ref + self.attitude.apply(self.pivot)
    self.initial_outer = self.r_max

  def observe(self, data, touch, tick):
    super().observe(data, touch, tick)
    self.tick = tick
    if self.face_ids is not None:
      self.strips = self.frontiers
      # The original grasp distance is a proposal, not a workspace boundary
      # on a different side of an elongated object.
      self.r_max = self.initial_outer + 0.12
    palm = float(touch.loads[-1])
    hits = touch.measured_heads
    delta = data.qpos[:3] - self.actual_position
    self.actual_position = data.qpos[:3].copy()
    self.recent_motion = 0.78 * self.recent_motion + 0.22 * delta
    self.empty = 0 if hits.any() else self.empty + 1
    if hits.sum() >= 2 and touch.loads.max() < 4 and palm < 0.35:
      self.saved_pose = (self.actual_position.copy(), rotation(data.qpos[3:7]))
      self.saved_tick = tick
    rising = palm > 0.7 and palm - self.force_before > 0.65
    self.force_before = palm
    if (palm > 1.8 or rising) and not self.clearance_ticks:
      radial = unit(self.actual_position - self.origin)
      back = (
        radial
        if np.linalg.norm(self.recent_motion) < 0.00008
        else -unit(self.recent_motion)
      )
      normal = (
        self.goal_normal
        if self.face_ready and self.goal_normal @ radial > 0.25
        else radial
      )
      self.outward = unit(0.48 * radial + 0.34 * normal + 0.30 * back)
      self.clearance_ticks = 1
      self.clear_frames = 0
      self.relief_count += 1
      self.last_palm_contact = self.actual_position.copy()
      self.rescue_until = 0
    elif self.clearance_ticks:
      self.clearance_ticks += 1
      self.clear_frames = self.clear_frames + 1 if palm < 0.4 else 0
      if self.clear_frames >= 8:
        self.clearance_ticks = 0
    if (
      self.empty >= 50
      and not self.clearance_ticks
      and self.saved_pose is not None
      and tick - self.saved_tick < 190
      and tick - self.last_rescue > 260
      and np.linalg.norm(self.saved_pose[0] - self.actual_position) < 0.075
    ):
      self.rescue_until, self.last_rescue = tick + 50, tick
      self.recontact_count += 1

  def plan(self, data, touch, tick):
    row = super().plan(data, touch, tick)
    surface = self.surface
    self.palm_planes = None
    if surface.tree is not None:
      hull = (
        self.palm_vertices @ rotation(data.qpos[3:7]).as_matrix().T
        + data.qpos[:3]
        - self.origin
      )
      distances, ids = surface.tree.query(hull)
      keep = distances < 0.040
      if keep.any():
        self.palm_planes = (
          self.palm_vertices[keep],
          surface.points[ids[keep]] + self.origin,
          surface.normals[ids[keep]],
        )
    row.update(
      palm_reliefs=self.relief_count,
      wrist_recontacts=self.recontact_count,
      axial_lanes=self.frontiers.lanes is not None,
    )
    return row

  def sync(self, point, attitude):
    self.attitude = attitude
    self.position_ref = point - self.origin
    unshifted = self.position_ref - self.shift
    self.radius, self.radial = np.linalg.norm(unshifted), unit(unshifted)

  def phase(self):
    if self.clearance_ticks:
      return "withdraw"
    if self.tick < self.rescue_until and self.empty:
      return "approach"
    return super().phase()

  def command(self, touch):
    pos, attitude = self.last_command, self.last_rotation
    palm_at_risk = (
      self.last_palm_contact is not None
      and np.linalg.norm(pos - self.last_palm_contact) < 0.027
    )
    if self.clearance_ticks:
      speed = min(0.17, 0.070 + 0.012 * touch.loads[-1])
      point = pos + DT * speed * self.outward
      self.sync(point, attitude)
      cmd = np.r_[point, quaternion(attitude)]
      ahead = cmd.copy()
    elif self.tick < self.rescue_until and self.empty and touch.loads.max() < 3:
      target, goal = self.saved_pose
      point = pos + limit(target - pos, 0.0009)
      turn = limit((goal * attitude.inv()).as_rotvec(), 0.009)
      attitude = Rotation.from_rotvec(turn) * attitude
      self.sync(point, attitude)
      cmd = np.r_[point, quaternion(attitude)]
      ahead = cmd.copy()
    else:
      cmd, ahead = super().command(touch)
      if self.palm_planes is not None:
        vertices, anchors, normals = self.palm_planes
        hull = rotation(cmd[3:]).apply(vertices) + cmd[:3]
        delta = hull - anchors
        gap = np.sum(delta * normals, axis=1)
        tangent = np.linalg.norm(delta - gap[:, None] * normals, axis=1)
        palm_at_risk |= bool(np.any((tangent < 0.024) & (gap > -0.015) & (gap < 0.012)))
        current_gap = np.sum(
          (attitude.apply(vertices) + pos - anchors) * normals, axis=1
        )
        valid = (tangent < 0.023) & (gap < 0.006) & (gap < current_gap)
        if valid.any():
          i = int(np.argmin(np.where(valid, gap, np.inf)))
          correction = min(0.0017, current_gap[i] - gap[i] + 0.00016) * normals[i]
          cmd[:3] = pos + limit(cmd[:3] + correction - pos, 0.002)
          ahead[:3] += correction
          self.sync(cmd[:3], rotation(cmd[3:]))
    # Slow an unsupported approach near a measured surface or a previous palm
    # impact. Away from those regions the fingers can still catch the moving target.
    # Without a coherent footprint, a distant mapped palm plane gives no
    # evidence of clearance at the next wrist orientation.
    blind_limit = palm_at_risk or not self.face_ready
    if self.empty >= 4 and blind_limit and not self.clearance_ticks:
      attitude = self.last_rotation
      wanted = rotation(cmd[3:])
      movement = (
        wanted.apply(self.palm_vertices)
        - attitude.apply(self.palm_vertices)
        + cmd[:3]
        - pos
      )
      distance = np.linalg.norm(movement, axis=1).max()
      if distance > 0.00080:
        fraction = 0.00080 / distance
        point = pos + fraction * (cmd[:3] - pos)
        wanted = (
          Rotation.from_rotvec(fraction * (wanted * attitude.inv()).as_rotvec())
          * attitude
        )
        cmd = np.r_[point, quaternion(wanted)]
        ahead[:3] = point + fraction * (ahead[:3] - point)
        future = rotation(ahead[3:])
        ahead[3:] = quaternion(
          Rotation.from_rotvec(fraction * (future * wanted.inv()).as_rotvec()) * wanted
        )
        self.sync(point, wanted)
    self.last_command, self.last_rotation = cmd[:3].copy(), rotation(cmd[3:])
    return cmd, ahead


class FrontierWorker(FaceWorker):
  command_fields = (*FaceWorker.command_fields, "palm_planes")
