"""Execute information-selected wrist motions with a causal tactile force servo."""

from collections import deque
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.active_gpis import ObservedSurface
from ..acitive_02.aude import unit
from ..acitive_02.surface_scan import limit
from ..i04_geometric_traversal.physics import quaternion, rotation
from .informative_actions import InformativeActions
from .contact_trail import ContactTrail
from .sweep import DT, LOAD_LIMIT, PLAN_TICKS, PREVIEW_S
from .wrist_worker import WristWorker


class InformativeWrist:
  def __init__(self, hand, data, touch, **options):
    if hand.objects or not options.get("gpis", True):
      raise ValueError("Expected an object-free planning hand and tactile GPIS")
    self.hand = hand
    self.surface = ObservedSurface()
    self.selector = InformativeActions.from_default_history()
    self.position = data.qpos[:3].copy()
    self.attitude = rotation(data.qpos[3:7])
    self.workspace_center = self.position.copy()
    self.goal = self.position.copy()
    self.goal_attitude = self.attitude
    self.measured_position = self.position.copy()
    self.recent_motion = np.zeros(3)
    self.normal = unit(touch.normals[touch.measured_heads].mean(axis=0))
    self.last_normals = touch.normals.copy()
    self.offsets = self.pad_offsets(data, touch)
    geom = hand.palm_geom
    model = hand.model
    mesh = model.geom_dataid[geom]
    begin = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[begin:begin + model.mesh_vertnum[mesh]]
    directions = np.array([[x,y,z] for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1)])
    vertices = vertices[np.unique(np.argmax(vertices @ directions.T, axis=0))]
    world = vertices @ data.geom_xmat[geom].reshape(3,3).T + data.geom_xpos[geom]
    self.palm = self.attitude.inv().apply(world-self.position)
    self.planes = None
    self.tick = self.empty = self.escape_count = 0
    self.last_palm_load = 0.
    self.unloading = False
    self.clear_ticks = 0
    self.relief = self.normal
    self.saved = None
    self.saved_tick = self.apply_tick = 0
    self.last_touch = self.rescue = None
    self.rescue_tick = self.good_ticks = 0
    self.anchors = deque(maxlen=96)
    self.trail = ContactTrail()
    self.return_waypoints = deque()
    self.trail_transfers = self.trail_recontacts = 0
    self.used_trail_this_loss = False
    self.tried_anchors = set()
    self.rescue_stage = "near"
    self.rescue_deadline = 0
    self.plan_mode = "scan"
    # Common trace writer reports completed periodic passes for its other policies.
    self.route = SimpleNamespace(passes=0)

  def pad_offsets(self, data, touch):
    pads = self.hand.pads
    return np.einsum("fji,fj->fi", data.geom_xmat[pads].reshape(-1,3,3),
                     touch.points-data.geom_xpos[pads])

  def transfer_to_contact(self, pos, tick):
    """Re-enter at a different previously measured light, multi-pad posture."""
    eligible = [(stamp, pose) for stamp, pose in self.anchors
                if np.linalg.norm(pose[0] - pos) <= .16]
    fresh = [(stamp, pose) for stamp, pose in eligible if stamp not in self.tried_anchors]
    farther = [(stamp, pose) for stamp, pose in fresh if np.linalg.norm(pose[0] - pos) > .018]
    if farther:
      def cost(item):
        point, orient, normal = item[1]
        return (np.linalg.norm(point-pos) + .025*(1-normal @ self.last_touch[2])
                + .015*(orient*self.attitude.inv()).magnitude())
      stamp, self.rescue = min(farther, key=cost)
      self.tried_anchors.add(stamp)
      self.return_waypoints = deque([self.rescue])
      distance = np.linalg.norm(self.rescue[0]-pos)
    else:
      route, distance = self.trail.return_path(pos, self.attitude, self.tried_anchors)
      self.rescue = route[-1] if route else self.last_touch
      self.return_waypoints = deque(route or [self.rescue])
      if len(route) > 1:
        self.trail_transfers += 1
        self.used_trail_this_loss = True
    self.rescue_stage = "clear"
    self.rescue_tick = tick
    self.rescue_deadline = tick + 260 + int(distance/.00050)
    self.escape_count += 1

  def observe(self, data, touch, tick):
    self.tick = tick
    hits = touch.measured_heads
    peak = float(touch.loads.max())
    if peak < LOAD_LIMIT:
      self.surface.add(touch, tick)
      if hits.any():
        self.offsets[hits] = self.pad_offsets(data,touch)[hits]
        self.last_normals[hits] = touch.normals[hits]
    self.empty = 0 if hits.any() else self.empty + 1
    pos = data.qpos[:3].copy()
    self.recent_motion = 0.78*self.recent_motion + 0.22*(pos-self.measured_position)
    self.measured_position = pos
    palm = float(touch.loads[-1])
    if hits.any() and peak < LOAD_LIMIT and palm < .9:
      n = unit(touch.normals[hits].mean(axis=0))
      self.last_touch = (pos, rotation(data.qpos[3:7]), n)
    if hits.sum() >= 2 and peak < 4 and palm < 0.35:
      self.saved = self.last_touch
      self.saved_tick = tick
      if not self.anchors or np.linalg.norm(pos-self.anchors[-1][1][0]) >= .018:
        self.anchors.append((tick, self.saved))
    outward = self.last_touch[2] if self.last_touch is not None else self.normal
    pose = (pos, rotation(data.qpos[3:7]), outward)
    self.trail.observe(tick, pose, int(hits.sum()) if peak < LOAD_LIMIT and palm < .9 else 0)
    if not self.unloading and (palm > 1.8 or (palm > 0.7 and palm-self.last_palm_load > 0.65) or peak >= 16):
      measured = unit(touch.normals[hits].mean(axis=0)) if hits.any() else self.normal
      back = -unit(self.recent_motion) if np.linalg.norm(self.recent_motion)>0.00008 else measured
      self.relief = unit(0.72*measured + 0.28*back)
      self.unloading = True
      self.clear_ticks = self.good_ticks = 0
      self.rescue = None
      self.escape_count += 1
    if self.unloading:
      self.clear_ticks = self.clear_ticks+1 if palm < 0.4 and peak < 5 else 0
      if self.clear_ticks >= 8:
        self.unloading = False
    if self.empty >= 18 and self.last_touch is not None and self.rescue is None and not self.unloading:
      nearby = self.saved is not None and tick-self.saved_tick <= 90 and np.linalg.norm(self.saved[0]-pos) < .035
      self.rescue = self.saved if nearby else self.last_touch
      self.rescue_tick, self.good_ticks = tick, 0
      self.rescue_stage = "near"
      self.rescue_deadline = tick + 85
      self.trail.freeze()
      self.used_trail_this_loss = False
      self.tried_anchors.clear()
      self.escape_count += 1
    if self.rescue is not None:
      self.good_ticks = self.good_ticks+1 if hits.any() and peak < LOAD_LIMIT and palm < .9 else 0
      if self.good_ticks >= 3:
        self.trail_recontacts += int(self.used_trail_this_loss)
        self.rescue = None
        self.goal, self.goal_attitude = self.position.copy(), self.attitude
      elif tick >= self.rescue_deadline or (self.rescue_stage == "near" and self.empty >= 40 and peak > 3.):
        self.transfer_to_contact(pos, tick)
    self.last_palm_load = palm

  def phase(self):
    if self.unloading:
      return "withdraw"
    if self.rescue is not None or (self.empty and self.plan_mode == "recontact"):
      return "approach"
    return "scan"

  def command(self, touch):
    peak = float(touch.loads.max())
    hits = touch.measured_heads
    old_pos, old_rot = self.position, self.attitude
    if self.unloading:
      step = DT * min(0.17, 0.070 + 0.012*peak) * self.relief
      turn = np.zeros(3)
    elif self.rescue is not None:
      point, attitude, normal = self.rescue
      age = self.tick - self.rescue_tick
      if self.rescue_stage == "clear" and age >= 22:
        self.rescue_stage = "transfer"
      if self.rescue_stage == "transfer":
        while self.return_waypoints:
          point, attitude, normal = self.return_waypoints[0]
          elevated = self.workspace_center + limit(point+.015*normal-self.workspace_center, .45)
          if np.linalg.norm(elevated-old_pos) >= .006 or (attitude*old_rot.inv()).magnitude() >= .18:
            break
          self.return_waypoints.popleft()
        if not self.return_waypoints:
          self.rescue_stage, self.rescue_tick = "approach", self.tick
          age = 0
      if self.rescue_stage == "clear":
        step = .0008*self.last_touch[2]
      elif self.rescue_stage == "transfer":
        step = limit(elevated-old_pos, .001)
      else:
        depth = min(.019, max(0, age-12)*.0003)
        step = limit(point-depth*normal-old_pos, .0009)
      if not hits.any() and peak > 3. and self.rescue_stage != "clear":
        # The finger link can be loaded while all tactile pads read zero.
        # Release it; holding this pose would make recontact stall indefinitely.
        step += min(.00095, (peak-3.)*.00016)*normal
      turn = limit((attitude*old_rot.inv()).as_rotvec(), .009)
    elif not hits.any():
      normal = self.last_touch[2] if self.last_touch is not None else self.normal
      remaining = max(1, self.apply_tick + PLAN_TICKS - self.tick)
      tangent = self.goal-old_pos
      tangent -= (tangent @ normal)*normal
      step = limit(tangent/remaining, .00045) + (.0006 if peak > 3 else -.0005)*normal
      turn = limit((self.goal_attitude*old_rot.inv()).as_rotvec()/remaining, .004)
    else:
      normal = self.normal
      if hits.any():
        measured = unit(touch.normals[hits].mean(axis=0))
        if measured @ normal > 0.2:
          normal = unit(0.64*normal + 0.36*measured)
      # Follow the winning segment in all three directions. Independently react to
      # measured force at servo rate; a fresh plan supersedes the old endpoint.
      remaining = max(1, self.apply_tick + PLAN_TICKS - self.tick)
      step = (self.goal-old_pos) / remaining
      turn = limit((self.goal_attitude*old_rot.inv()).as_rotvec()/remaining, 0.0125)
      if peak > 4.:
        correction = min(0.12, 0.013*(peak-3.0))
      elif hits.sum() < 2:
        correction = -0.060 if peak < 1.3 else -0.014
      elif hits.sum() == 2:
        correction = -0.012 if peak < 2.6 else -0.002
      else:
        correction = np.clip(0.006*(peak-1.6),-0.008,0.009)
      step = limit(step,0.0014) + DT*correction*normal
    attitude = Rotation.from_rotvec(turn)*old_rot
    step = limit(step,0.002)
    # Use only measured surface tangent planes around the palm hull.
    palm_risk = False
    if self.planes is not None and not self.unloading:
      verts, anchors, normals = self.planes
      delta = attitude.apply(verts) + old_pos + step - anchors
      gap = np.sum(delta*normals,axis=1)
      lateral = np.linalg.norm(delta-gap[:,None]*normals,axis=1)
      previous = np.sum((old_rot.apply(verts)+old_pos-anchors)*normals,axis=1)
      palm_risk = np.any((lateral<0.024) & (gap>-.015) & (gap<.012))
      closing = (lateral<.023) & (gap<.006) & (gap<previous)
      if closing.any():
        j = np.argmin(np.where(closing,gap,np.inf))
        step = limit(step + min(.0017, previous[j]-gap[j]+.00016)*normals[j], .002)
    if self.empty >= 4 and not self.unloading:
      motion = attitude.apply(self.palm) - old_rot.apply(self.palm) + step
      bound = 0.0008 if palm_risk or self.empty >= 20 else 0.0016
      fraction = min(1., bound/max(np.linalg.norm(motion,axis=1).max(),1e-12))
      step *= fraction
      attitude = Rotation.from_rotvec(fraction*turn)*old_rot
    position = old_pos+step
    self.position = self.workspace_center + limit(position-self.workspace_center, .45)
    self.attitude = attitude
    cmd = np.r_[self.position,quaternion(attitude)]
    if self.unloading or self.rescue is not None:
      return cmd,cmd.copy()
    future = attitude * Rotation.from_rotvec((old_rot.inv()*attitude).as_rotvec()*PREVIEW_S/DT)
    ahead = np.r_[self.position+PREVIEW_S/DT*(self.position-old_pos),quaternion(future)]
    return cmd,ahead


class AppendOnlySurface(ObservedSurface):
  """The first measurement in each voxel is immutable and indices are append-only."""
  def refresh(self):
    if len(self.samples) != len(self.points):
      retained = self.patches
      self.patches = {}
      super().refresh()
      if len(retained) > 240:
        retained = dict(list(retained.items())[-160:])
      self.patches = retained


class InformativeWorker(WristWorker):
  """Hand the acquisition worker immutable sensor snapshots, including tactile deltas."""
  command_fields = ("goal","goal_attitude","normal","planes","plan_mode")

  def __init__(self, servo, paced):
    super().__init__(servo, paced)
    self.map = AppendOnlySurface()

  def submit(self, data, touch, tick, lead):
    servo = self.servo
    wrist = data.qpos[:3].copy()
    orient = rotation(data.qpos[3:7])
    pads = servo.hand.pads
    heads = data.geom_xpos[pads] + np.einsum("fij,fj->fi",data.geom_xmat[pads].reshape(-1,3,3),servo.offsets)
    verts = orient.apply(servo.palm) + wrist
    q, hits = data.qpos[servo.hand.q].copy(), touch.measured_heads.copy()
    normals, loads = servo.last_normals.copy(), touch.loads.copy()
    delta,self.delta = self.delta,{}
    last = servo.surface.last_observation_tick
    observed_phase = servo.phase()
    transfers, recontacts = servo.trail_transfers, servo.trail_recontacts

    def compute():
      began = perf_counter()
      self.map.samples.update(delta)
      self.map.last_observation_tick = last
      self.map.refresh()
      result = servo.selector.select(surface=self.map, q=q, hits=hits, normals=normals, loads=loads,
          wrist_position=wrist,wrist_rotation=orient,pads=heads,palm_vertices=verts,tick=tick,
          workspace_center=servo.workspace_center, observed_phase=observed_phase)
      goal, desired = result["position"],result["rotation"]
      normal = unit(result["normal"])
      if not np.isfinite(np.r_[goal,desired.as_rotvec(),normal]).all() or np.linalg.norm(normal)<.9:
        raise ValueError("Invalid information-selected wrist reference")
      planes = None
      if self.map.tree is not None:
        distances,indices = self.map.tree.query(verts)
        near = distances < .040
        if near.any():
          ids = indices[near]
          planes = (servo.palm[near],self.map.points[ids],self.map.normals[ids])
      info = dict(result["diagnostics"])
      mode = info.get("mode", result.get("mode","explore"))
      model = servo.selector.global_model
      info.update(tick=tick,cells=len(self.map.samples),last_new_tick=last,
                  trail_transfers=transfers,trail_recontacts=recontacts,
                  global_fit_tick=model.last_fit_tick,global_contact_cells=model.count,
                  global_queries=len(model.query) if model.count else 0,
                  observed_phase=observed_phase,supported=int(hits.sum()),
                  selected_direction=np.asarray(result["direction"]).tolist(),
                  wrist_target_m=np.asarray(goal).tolist(),wrist_target_xyzw=desired.as_quat().tolist(),
                  planning_ms=1000*(perf_counter()-began))
      return info,(np.asarray(goal),desired,normal,planes,mode)
    self.due = tick+lead
    self.pending = self.pool.submit(compute)

  def install(self, tick, *, initial=False):
    row = super().install(tick,initial=initial)
    if row is not None:
      self.servo.apply_tick = tick
      row["application_phase"] = self.servo.phase()
    return row
