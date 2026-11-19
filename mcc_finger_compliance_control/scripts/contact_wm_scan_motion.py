"""Seeded, nonperiodic object motion with continuous position/velocity/acceleration.

This constructs object commands only, never finger joint references. A fixed
base-time program is sampled at t*rate, so speed randomization cannot change
the geometry of a paired command. Physical feasibility still needs auditing.
"""
import numpy as np
from scipy.interpolate import BPoly, PPoly


def norm_extrema(poly, coordinates):
    """Evaluate endpoints and stationary points of each squared vector norm.

    Normalize each segment to [0,1] before polynomial root finding, avoiding
    powers of the absolute trajectory time. No sampled clipping is involved.
    """
    pp=PPoly.from_bernstein_basis(poly)
    maximum=0.
    for i,dt in enumerate(np.diff(pp.x)):
        coefficients=pp.c[::-1,i,coordinates]*dt**np.arange(len(pp.c))[:,None]
        squared=sum(np.polynomial.polynomial.polymul(c,c) for c in coefficients.T)
        roots=np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(squared))
        candidates=[0.,1.]+[float(x.real) for x in roots if abs(x.imag)<1e-7 and 0<x.real<1]
        values=np.polynomial.polynomial.polyval(candidates,coefficients)
        maximum=max(maximum,float(np.linalg.norm(values,axis=0).max()))
    return maximum*(1+1e-10)


class CompoundScan:
    def __init__(self, seed, base_duration_s=120., knot_count=25,
                 translation_radius_m=.07, rotation_radius_rad=.22,
                 base_speed_m_s=.025, base_angular_speed_rad_s=.08):
        if base_duration_s <= 0 or knot_count < 4:
            raise ValueError('Positive duration and at least four knots required')
        limits=np.array([translation_radius_m,rotation_radius_rad,base_speed_m_s,base_angular_speed_rad_s])
        if not np.isfinite(limits).all() or np.any(limits<=0):
            raise ValueError('Positive finite motion bounds required')
        rng=np.random.default_rng(seed)
        intervals=rng.uniform(4.,7.,knot_count-1)
        self.times=np.r_[0.,np.cumsum(intervals)]
        self.times*=base_duration_s/self.times[-1]
        values=np.zeros((knot_count,5))
        # Select new translations away from recent nodes. Revisit is possible,
        # but no cycle/ellipse is tiled to create a longer nominal trajectory.
        for i in range(1,knot_count):
            candidates=rng.uniform(-1.,1.,(12,2))*.055
            recent=values[max(0,i-6):i,:2]
            score=np.linalg.norm(candidates[:,None,:]-recent[None,:,:],axis=-1).min(-1)
            values[i,:2]=candidates[np.argmax(score)]
            values[i,2:]=rng.uniform(-.16,.16,3)
        velocity=np.zeros_like(values)
        velocity[1:-1]=.75*(values[2:]-values[:-2])/(self.times[2:]-self.times[:-2])[:,None]
        acceleration=np.zeros_like(values)
        spline=BPoly.from_derivatives(self.times,np.stack([values,velocity,acceleration],axis=1))
        # Include between-command extrema. Rescale the whole program, never
        # clip live steps. Control-polygon velocity bounds would shrink these
        # quintics excessively and erase the intended spatial coverage.
        coefficients=spline.c
        scale=[]
        for sl,radius,speed in [(slice(0,2),translation_radius_m,base_speed_m_s),
                                (slice(2,5),rotation_radius_rad,base_angular_speed_rad_s)]:
            scale.append(min(1.,radius/norm_extrema(spline,sl),
                             speed/norm_extrema(spline.derivative(),sl)))
        coefficients[...,:2]*=scale[0];coefficients[...,2:]*=scale[1]
        self.spline=spline
        self.base_duration_s=float(base_duration_s)
        self.metadata=dict(seed=int(seed),kind='nonperiodic_C2_compound_scan',
            base_duration_s=self.base_duration_s,knot_times_s=self.times.tolist(),
            knot_values_xy_rotvec=spline(self.times).tolist(),
            knot_velocities_xy_rotvec=spline.derivative()(self.times).tolist(),
            translation_scale=scale[0],rotation_scale=scale[1],
            translation_radius_m=translation_radius_m,rotation_radius_rad=rotation_radius_rad,
            base_speed_bound_m_s=base_speed_m_s,base_rotvec_speed_bound_rad_s=base_angular_speed_rad_s,
            no_finger_waypoints=True,no_contact_feedback=True)

    def evaluate(self, physical_time_s, rate=1., derivative=0):
        if not np.isfinite(rate) or rate<=0:
            raise ValueError('Positive finite speed multiplier required')
        t=np.asarray(physical_time_s,dtype=float)*rate
        if not np.isfinite(t).all() or np.any(t<0) or np.any(t>self.base_duration_s):
            raise ValueError('Outside the published motion program; no silent wrap or restart')
        return self.spline(t,nu=derivative)*rate**derivative

    def sampled_report(self, duration_s, rate, dt=.01):
        t=np.arange(int(round(duration_s/dt)))*dt
        q=self.evaluate(t,rate);v=self.evaluate(t,rate,1);a=self.evaluate(t,rate,2)
        voxels=np.floor(q[:,:2]/.005).astype(int)
        unique=len(np.unique(voxels,axis=0))
        return dict(duration_s=duration_s,motion_rate=rate,frames=len(t),
            translation_xy_span_m=np.ptp(q[:,:2],axis=0).tolist(),
            translation_path_m=float(np.linalg.norm(np.diff(q[:,:2],axis=0),axis=-1).sum()),
            max_translation_speed_m_s=float(np.linalg.norm(v[:,:2],axis=-1).max()),
            max_translation_acceleration_m_s2=float(np.linalg.norm(a[:,:2],axis=-1).max()),
            max_rotvec_speed_rad_s=float(np.linalg.norm(v[:,2:],axis=-1).max()),
            unique_command_xy_cells_5mm=unique,
            coverage_note='Command-space cells only; actual loaded object-surface coverage must be measured after physics')
