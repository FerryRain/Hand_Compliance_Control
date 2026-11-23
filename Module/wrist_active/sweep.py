"""Tactile GPIS exploration using a wrist orbit around the known fixture origin."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..acitive_02.active_gpis import ObservedSurface
from ..acitive_02.aude import unit
from ..acitive_02.surface_scan import align, limit
from ..i04_geometric_traversal.physics import quaternion, rotation


DT = .01
PLAN_TICKS = 20
PREVIEW_S = .20
ANGULAR_SPEED = .32
ROW_PITCH = .48
LOAD_LIMIT = 8.


class Spiral:
    """Continuous, approximately constant-speed arcs from one pole to its antipode."""

    def __init__(self, radial, side):
        self.axis = unit(radial)
        self.side = unit(side - (side @ self.axis) * self.axis)
        self.angle = 0.
        self.passes = 0
        self.pitch = ROW_PITCH

    def at(self, angle, phase=0.):
        phase = (2 * np.pi / self.pitch) * angle + phase
        return (np.cos(angle)[..., None] * self.axis + np.sin(angle)[..., None] *
                (np.cos(phase)[..., None] * self.side +
                 np.sin(phase)[..., None] * np.cross(self.axis, self.side)))

    def advance(self, arc, points):
        if self.angle >= np.pi - 1e-9:
            self.axis = -self.axis
            self.angle = 0.
            self.passes += 1
            self.pitch = max(.30, ROW_PITCH - .055 * self.passes)
            # Choose the next sweep's azimuth from gaps in actual contact directions.
            phases = (np.arange(10) + .5) * (2*np.pi/10)
            phase = phases[self.passes % len(phases)]
            if len(points):
                radius = np.linalg.norm(points, axis=1)
                tree = cKDTree(points / np.maximum(radius[:, None], 1e-6))
                angles = np.linspace(.12, np.pi-.12, 96)
                scores = [np.mean(np.minimum(tree.query(self.at(angles, a))[0], .34)
                                  * np.sin(angles)) for a in phases]
                phase = phases[np.argmax(scores)]
            self.side = Rotation.from_rotvec(self.axis*phase).apply(self.side)
        # Midpoint arc-length parameterization also behaves smoothly near the poles.
        frequency = 2*np.pi/self.pitch
        mid = self.angle + .5*arc / np.hypot(1., frequency*np.sin(self.angle))
        self.angle = min(np.pi, self.angle + arc/np.hypot(1., frequency*np.sin(mid)))
        return self.at(self.angle)


class ContactPrior:
    """A regularized quadratic implicit mean, fitted to measured points and normals."""

    def fit(self, points, normals):
        selected = np.linspace(0,len(points)-1,min(len(points),240),dtype=int)
        pts, n = points[selected],normals[selected]
        self.scale = np.median(np.linalg.norm(pts,axis=1))
        x,y,z = (pts/self.scale).T
        features = np.stack((x*x,y*y,z*z,2*x*y,2*x*z,2*y*z,x,y,z),axis=1)
        derivative = np.zeros((len(x),3,9))
        derivative[:,0,0], derivative[:,1,1], derivative[:,2,2] = 2*x,2*y,2*z
        derivative[:,0,3], derivative[:,1,3] = 2*y,2*x
        derivative[:,0,4], derivative[:,2,4] = 2*z,2*x
        derivative[:,1,5], derivative[:,2,5] = 2*z,2*y
        derivative[:,:,6:] = np.eye(3)
        tangent = (np.eye(3)-n[:,:,None]*n[:,None,:]) @ derivative
        weight = .24*np.sqrt(min(25,len(x)))
        prior = np.array([1.,1.,1.,0.,0.,0.,0.,0.,0.])
        design = np.vstack((features,.34*tangent.reshape(-1,9),weight*np.eye(9)))
        rhs = np.r_[np.ones(len(x)),np.zeros(3*len(x)),weight*prior]
        a,b,c,d,e,f,*linear = np.linalg.lstsq(design,rhs,rcond=None)[0]
        vals, vec = np.linalg.eigh([[a,d,e],[d,b,f],[e,f,c]])
        self.matrix = (vec*np.clip(vals,.18,6.))@vec.T
        self.linear = np.asarray(linear)

    def predict(self, points):
        x = points/self.scale
        hx = x@self.matrix
        gradient = 2*hx+self.linear
        magnitude = np.maximum(np.linalg.norm(gradient,axis=-1),.30)
        distance = self.scale*(np.sum(x*hx,axis=-1)+x@self.linear-1)/magnitude
        return distance, gradient/magnitude[...,None]


class SweepWrist:
    """The planner reads measured joints and tactile packets. Its only output is SE(3)."""

    def __init__(self, hand, data, touch, *, gpis=True, recontact=True):
        if hand.objects:
            raise ValueError("The wrist planner requires an object-free robot model")
        self.hand = hand
        self.gpis = gpis
        self.recontact = recontact
        self.surface = ObservedSurface()
        self.prior = ContactPrior()
        self.attitude = rotation(data.qpos[3:7])
        self.radius = np.linalg.norm(data.qpos[:3])
        self.radial = unit(data.qpos[:3])
        axes = self.attitude.as_matrix().T
        side = axes[np.argmin(np.abs(axes @ self.radial))]
        self.route = Spiral(self.radial, side)
        self.base_attitude = self.attitude
        self.twist = np.zeros(3)
        self.local_tilt = np.zeros(3)
        self.offsets = np.einsum('fji,fj->fi', data.geom_xmat[hand.pads].reshape(-1,3,3),
                                 touch.points-data.geom_xpos[hand.pads])
        self.last_normals = touch.normals.copy()
        self.r_min, self.r_max = self.radius-.10, self.radius+.13
        self.bad_ticks = 0
        self.overload_ticks = 0
        self.escape_ticks = 0
        self.escape_count = 0
        self.last_escape = -1000
        self.feed_speed = 0.
        self.radial_speed = 0.
        self.shift = np.zeros(3)
        self.shift_velocity = np.zeros(3)

    def observe(self, data, touch, tick):
        light = touch.loads.max() < LOAD_LIMIT
        hits = touch.measured_heads
        if light:
            self.surface.add(touch, tick)
            if hits.any():
                ids = self.hand.pads[hits]
                self.offsets[hits] = np.einsum('fji,fj->fi', data.geom_xmat[ids].reshape(-1,3,3),
                                              touch.points[hits]-data.geom_xpos[ids])
                self.last_normals[hits] = touch.normals[hits]
        self.bad_ticks = 0 if light and hits.sum() >= 2 else self.bad_ticks + 1
        self.overload_ticks = self.overload_ticks+1 if not light else 0
        age = tick-(self.surface.last_observation_tick or 0)
        if self.escape_ticks:
            self.escape_ticks -= 1
        elif self.recontact and tick-self.last_escape >= 480 and (self.overload_ticks >= 65 or
                (self.bad_ticks >= 260 and age >= 95) or age >= 650):
            self.escape_ticks = 100
            self.bad_ticks = 0
            self.escape_count += 1
            self.last_escape = tick

    def phase(self):
        if self.escape_ticks > 72:
            return 'withdraw'
        if self.escape_ticks > 40:
            return 'transfer'
        return 'approach' if self.escape_ticks else 'scan'

    def plan(self, data, touch, tick):
        if tick-getattr(self.surface,'fit_tick',-1000) >= 95 or self.surface.tree is None:
            self.surface.refresh()
            self.surface.fit_tick = tick
            if self.gpis:
                self.prior.fit(self.surface.points,self.surface.normals)
        light = touch.loads.max() < LOAD_LIMIT
        hits = touch.measured_heads
        speed = ANGULAR_SPEED if light and hits.sum() >= 2 else .19
        if self.escape_ticks:
            speed = .40
        step = PLAN_TICKS * DT
        goal = self.route.advance(speed*step, self.surface.points)
        turn = align(self.radial, goal)
        # The scan progresses even with one valid pad. Load controls radial depth
        # at 100 Hz and cannot pin the route index to the last two-pad pose.
        orbit = Rotation.from_rotvec(limit(turn.as_rotvec(), speed*step*1.15))
        next_radial = orbit.apply(self.radial)
        next_base = orbit*self.base_attitude
        live = rotation(data.qpos[3:7])
        pads = data.geom_xpos[self.hand.pads] + np.einsum(
            'fij,fj->fi', data.geom_xmat[self.hand.pads].reshape(-1,3,3), self.offsets)
        local = live.inv().apply(pads-data.qpos[:3])
        center = local.mean(axis=0)
        old = .995*self.local_tilt
        prior_ori = next_base*Rotation.from_rotvec(old)
        nominal = next_radial*self.radius
        pivot = nominal+next_base.apply(center)
        lever = prior_ori.apply(local-center)
        tips = pivot+lever
        gap = np.zeros(4)
        dirs = np.broadcast_to(self.radial,(4,3)).copy()
        patches = []
        if self.gpis:
            gap, dirs = self.prior.predict(tips)
            now_gap = self.prior.predict(pads)[0]
            gap[hits] -= now_gap[hits]
            for i, point in enumerate(pads):
                patch = self.surface.patch(point,preferred_normal=self.last_normals[i] if hits[i] else dirs[i])
                patches.append(patch)
                if patch is not None:
                    val, grad = patch.predict(tips[i])
                    mag = max(np.linalg.norm(grad[0]),.5)
                    estimate = (val[0] - (patch.predict(point)[0][0] if hits[i] else 0.))/mag
                    weight = np.clip(1.-np.linalg.norm(patch.anchor-point)/.045,0.,1.)
                    gap[i] = weight*estimate+(1.-weight)*gap[i]
                    dirs[i] = unit(weight*grad[0]/mag+(1.-weight)*dirs[i])
        # Each triplet yields one closed-form wrist height and tilt correction.
        # The 3 variables are wrist height (m) and two tilts in length units.
        # Joint angles are measured and held fixed for all five candidates.
        axes = next_base.as_matrix()[:,:2]
        rows = np.column_stack((dirs@next_radial, np.cross(lever,dirs)@axes/.065))
        support = np.where(hits,1.,.58)
        desired = np.clip(-gap-.0007,-.023,.023)
        desired += .00045*np.clip(touch.loads[:4]-1.2,-1,9)
        shifts = [np.zeros(3)]
        for drop in range(4):
            w = support.copy()
            w[drop] *= .13
            system = w[:,None]*rows
            shifts.append(np.linalg.solve(system.T@system+.055*np.eye(3),system.T@(w*desired)))
        shifts = np.asarray(shifts)
        heights = np.clip(shifts[:,0],-.006,.006)
        choices = np.asarray([limit(old+np.r_[limit(v[1:]/.065,.105),0.],.76) for v in shifts])
        orientations = next_base*Rotation.from_rotvec(choices)
        pivot_shift = next_base.apply(center)-orientations.apply(np.broadcast_to(center,choices.shape))
        predicted = nominal + pivot_shift[:,None] + heights[:,None,None]*next_radial
        predicted = predicted+np.einsum('aij,fj->afi',orientations.as_matrix(),local)
        predicted_gap = gap+np.einsum('afi,fi->af',predicted-tips,dirs)
        probability = np.exp(-.5*((predicted_gap+.0007)/.006)**2)
        ranked = np.sort(probability,axis=1)
        info = np.zeros_like(probability)
        if self.gpis:
            projected = predicted - predicted_gap[:,:,None]*dirs
            novelty = np.minimum(1.,self.surface.tree.query(projected)[0]/.010)
            for i, field in enumerate(patches):
                variance = field.predict(projected[:,i],variance=True) if field is not None else .75
                info[:,i] = novelty[:,i]*np.sqrt(variance)
        utility = (ranked[:,-2] + 1.2*ranked[:,-3]+.25*ranked[:,-4] +
                   .35*np.sum(probability*info,axis=1) -
                   .10*np.linalg.norm(choices-old,axis=1)/.105 -
                   .75*np.maximum(0.,-predicted_gap.min(axis=1)-.003)/.008)
        if not self.gpis:
            utility[1:] = -1.
        best = int(np.argmax(utility))
        self.feed_speed = float(.85*heights[best]/step)
        self.local_tilt = choices[best]
        self.base_attitude = next_base
        self.twist = orbit.as_rotvec()/step
        self.spin = (orientations[best]*self.attitude.inv()).as_rotvec()/step
        self.shift_velocity = (pivot_shift[best]-self.shift)/step
        return dict(tick=tick, pass_index=self.route.passes, arc=self.route.angle,
                    cells=len(self.surface.samples), last_new_tick=self.surface.last_observation_tick,
                    escape_count=self.escape_count, choice=best, supported=int(hits.sum()),
                    estimated_third=float(ranked[best,-3]), value=float(utility[best]))

    def command(self, touch):
        peak = touch.loads.max()
        hits = int(touch.measured_heads.sum())
        phase = self.phase()
        if phase == 'withdraw':
            velocity = .040
        elif phase == 'transfer' and peak < 4.:
            velocity = 0.
        elif peak > 5.:
            velocity = min(.14, .010*(peak-4.))
        elif hits < 2:
            velocity = -.052 if peak < 1.7 else -.010
        elif hits == 2:
            velocity = -.016 if peak < 2.6 else -.002
        else:
            velocity = np.clip(.004*(peak-1.8), -.008, .009)
        self.radial_speed = float(np.clip(velocity+self.feed_speed, -.06,.13))
        self.radius = float(np.clip(self.radius + DT*self.radial_speed, self.r_min,self.r_max))
        self.radial = Rotation.from_rotvec(self.twist*DT).apply(self.radial)
        self.attitude = Rotation.from_rotvec(self.spin*DT)*self.attitude
        self.shift += self.shift_velocity*DT
        command = np.r_[self.radius*self.radial+self.shift, quaternion(self.attitude)]
        future_radial = Rotation.from_rotvec(PREVIEW_S*self.twist).apply(self.radial)
        future_ori = Rotation.from_rotvec(PREVIEW_S*self.spin)*self.attitude
        future_r = np.clip(self.radius+PREVIEW_S*self.feed_speed, self.r_min,self.r_max)
        future_shift = self.shift + PREVIEW_S*self.shift_velocity
        return command, np.r_[future_r*future_radial+future_shift, quaternion(future_ori)]
