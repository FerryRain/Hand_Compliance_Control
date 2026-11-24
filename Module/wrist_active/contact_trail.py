"""Retrieve an earlier tactile posture along the wrist path that led to loss."""
from collections import deque

import numpy as np


class ContactTrail:
    def __init__(self):
        self.history = deque(maxlen=512)
        self.snapshot = ()

    def observe(self, tick, pose, light_fingers):
        p, orient, _ = pose
        if self.history:
            _, previous, quality = self.history[-1]
            if np.linalg.norm(p-previous[0]) < .009 and (orient*previous[1].inv()).magnitude() < .10:
                if light_fingers > quality:
                    self.history[-1] = (tick, pose, light_fingers)
                return
        self.history.append((tick, pose, light_fingers))

    def freeze(self):
        # Failed retries must not push the path to the last good contact out of memory.
        self.snapshot = tuple(self.history)

    def return_path(self, pos, attitude, tried):
        if not self.snapshot:
            return [], 0.
        stamps, poses, quality = zip(*self.snapshot)
        xyz = np.stack([p[0] for p in poses])
        quat = np.stack([p[1].as_quat() for p in poses])
        length = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        angles = 2*np.arccos(np.clip(abs(np.sum(quat[1:]*quat[:-1], axis=1)), 0., 1.))
        cumulative = np.r_[0., np.cumsum(length+.035*angles)]
        d = np.linalg.norm(xyz-pos, axis=1)
        orient_gap = 2*np.arccos(np.clip(abs(quat @ attitude.as_quat()), 0., 1.))
        entry = len(d)-1-int(np.argmin((d+.035*orient_gap)[::-1]))
        travel = abs(cumulative-cumulative[entry]) + d[entry] + .035*orient_gap[entry]
        quality = np.asarray(quality)
        eligible = (quality > 0) & (d > .018)
        if not eligible.any():
            return [], 0.
        fresh = eligible & ~np.isin(stamps, list(tried))
        if not fresh.any():
            tried.clear()
            fresh = eligible
        strong = fresh & (quality >= 2) & (travel <= .75)
        choose = strong if strong.any() else fresh
        target = int(np.argmin(np.where(choose, travel, np.inf)))
        tried.add(stamps[target])
        direction = 1 if target >= entry else -1
        return [poses[i] for i in range(entry, target+direction, direction)], float(travel[target])
