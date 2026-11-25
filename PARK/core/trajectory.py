"""Smoothly revise only future samples beyond all previously published previews."""
import numpy as np
from scipy.spatial.transform import Rotation

from .common import DT, pack, rot


def extrema(samples):
    if len(samples) < 2:
        return np.zeros(4)
    v = np.diff(samples[:, :3], axis=0)/DT
    orientations = rot(samples)
    omega = (orientations[1:]*orientations[:-1].inv()).as_rotvec()/DT
    def peak(a):
        return np.linalg.norm(a, axis=1).max(initial=0.)
    return np.array([peak(v), peak(omega), peak(np.diff(v, axis=0))/DT, peak(np.diff(omega, axis=0))/DT])


def limits(cfg):
    return np.array([cfg.linear_speed_m_s, cfg.angular_speed_rad_s, cfg.linear_accel_m_s2, cfg.angular_accel_rad_s2])


class Path:
    def __init__(self, start, samples):
        self.start = int(start)
        self.samples = np.atleast_2d(samples)
        self.end = self.start + len(self.samples) - 1

    def at(self, tick):
        return self.samples[np.clip(np.asarray(tick)-self.start, 0, len(self.samples)-1)]

    def revised(self, splice, goal, cfg):
        start = self.at(splice)
        distance = np.linalg.norm(goal[:3]-start[:3])
        angle = (rot(goal)*rot(start).inv()).magnitude()
        duration = max(cfg.min_segment_s, 1.88*distance/cfg.linear_speed_m_s, 1.88*angle/cfg.angular_speed_rad_s,
                       np.sqrt(5.8*distance/cfg.linear_accel_m_s2), np.sqrt(5.8*angle/cfg.angular_accel_rad_s2))
        for _ in range(9):
            steps = int(np.ceil(duration/DT))
            old = self.at(splice+np.arange(steps+1))
            t = np.linspace(0, 1, steps+1)
            blend = (t**3*(10-15*t+6*t*t))[:, None]
            before = rot(old)
            log = (rot(goal)*before.inv()).as_rotvec()
            samples = pack(old[:, :3] + blend*(goal[:3]-old[:, :3]), Rotation.from_rotvec(blend*log)*before)
            samples[0] = start
            flips = (samples[1:, 3:]*samples[:-1, 3:]).sum(1) < 0
            samples[1:, 3:] *= np.cumprod(np.where(flips, -1, 1))[:, None]
            extended = np.vstack((self.at([splice-2, splice-1]), samples, samples[-1]))
            if (extrema(extended) <= limits(cfg)*(1+1e-8)).all():
                return Path(splice, samples)
            duration *= 1.4
        return None
