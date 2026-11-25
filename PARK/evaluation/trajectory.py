"""A committed 100 Hz XYZ+wxyz wrist trajectory with a 200 ms forward sample."""
import numpy as np
from scipy.spatial.transform import Rotation

from .settings import DT, pose, rotation


class Trajectory:
    def __init__(self, start, samples):
        self.start = int(start)
        self.samples = np.atleast_2d(samples)

    @property
    def end(self):
        return self.start + len(self.samples) - 1

    def at(self, ticks):
        return self.samples[np.clip(np.asarray(ticks) - self.start, 0, len(self.samples) - 1)]

    def revise(self, splice, goal, cfg):
        base = self.at(splice)
        trans = np.linalg.norm(goal[:3] - base[:3])
        angle = (rotation(goal) * rotation(base).inv()).magnitude()
        duration = max(.60, 1.875 * trans / cfg.translation_speed_m_s,
                       1.875 * angle / cfg.rotation_speed_rad_s)
        count = int(np.ceil(duration / DT))
        t = np.arange(count + 1) / count
        blend = t**3 * (10 + t * (-15 + 6 * t))
        previous = self.at(splice + np.arange(count + 1))
        position = previous[:, :3] + blend[:, None] * (goal[:3] - previous[:, :3])
        before = rotation(previous)
        delta = (rotation(goal) * before.inv()).as_rotvec()
        orientations = Rotation.from_rotvec(blend[:, None] * delta) * before
        out = pose(position, orientations)
        # Use one continuous quaternion lift, including across +/- pi.
        signs = np.sign(np.einsum("ij,ij->i", out[1:, 3:], out[:-1, 3:]))
        out[1:, 3:] *= np.cumprod(np.where(signs < 0, -1., 1.))[:, None]
        out[0] = base
        return Trajectory(splice, out)

