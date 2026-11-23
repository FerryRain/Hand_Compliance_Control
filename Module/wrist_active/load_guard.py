"""A force-priority wrist supervisor for the existing tactile exploration planner."""
import numpy as np
from scipy.spatial.transform import Rotation

from ..i04_geometric_traversal.physics import quaternion
from .sweep import DT,PREVIEW_S,SweepWrist


class GuardedWrist(SweepWrist):
    """Hold tangential motion on a force excursion and unload above the sweep."""

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.braking=False
        self.clear_ticks=0
        self.brake_events=0

    def observe(self,data,touch,tick):
        super().observe(data,touch,tick)
        peak=float(touch.loads.max())
        if peak>=16. or touch.loads[-1]>=6.:
            self.brake_events+=int(not self.braking)
            self.braking=True
            self.clear_ticks=0
        elif self.braking:
            self.clear_ticks=self.clear_ticks+1 if peak<5. and touch.loads[-1]<2. else 0
            if self.clear_ticks>=5:
                self.braking=False
                # Replan from the held orientation once the load has settled.
                self.base_attitude=self.attitude*Rotation.from_rotvec(-self.local_tilt)
                self.twist=np.zeros(3)
                self.spin=np.zeros(3)
                self.shift_velocity=np.zeros(3)
                self.feed_speed=0.

    def plan(self,data,touch,tick):
        if self.braking:
            self.twist=self.spin=self.shift_velocity=np.zeros(3)
            self.feed_speed=0.
            return dict(tick=tick,pass_index=self.route.passes,arc=self.route.angle,
                        cells=len(self.surface.samples),last_new_tick=self.surface.last_observation_tick,
                        escape_count=self.escape_count,choice=-1,supported=int(touch.measured_heads.sum()),
                        estimated_third=0.,value=0.,load_brake=True,brake_events=self.brake_events)
        info=super().plan(data,touch,tick)
        return dict(**info,load_brake=False,brake_events=self.brake_events)

    def command(self,touch):
        peak=float(touch.loads.max())
        if self.braking:
            # Leave the measured load before resuming any lateral motion. This
            # overrides both the 28-tick withdrawal and the estimated GP slope.
            speed=np.clip(.018+.006*peak,.025,.16) if peak>=5. else 0.
            self.radial_speed=float(speed)
            self.radius=min(self.r_max,self.radius+DT*speed)
            now=np.r_[self.radius*self.radial+self.shift,quaternion(self.attitude)]
            forecast_radius=min(self.r_max,self.radius+min(PREVIEW_S,.05)*speed)
            return now,np.r_[forecast_radius*self.radial+self.shift,now[3:]]
        return super().command(touch)
