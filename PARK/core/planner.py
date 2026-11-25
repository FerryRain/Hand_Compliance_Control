"""The active decision is a single wrist transform scored over the uncertain zero set."""
from time import perf_counter

import numpy as np

from .belief import Belief
from .probe import Probe


def ms(start):
    return (perf_counter()-start)*1000


def scale(values):
    return (values-values.min())/max(float(np.ptp(values)), 1e-12)


class Planner:
    def __init__(self, models, cfg, variant, origin, plate):
        self.gp = Belief.load(models / "gpis.pt")
        self.probe = Probe(models, variant)
        self.cfg, self.variant, self.origin, self.plate = cfg, variant, origin, plate
