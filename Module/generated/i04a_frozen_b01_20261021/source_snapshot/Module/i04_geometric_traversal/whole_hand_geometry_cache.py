"""Reuse identical geometric queries within the unchanged dual command guard."""
from collections import OrderedDict
import numpy as np
from .sliding_safety import SHALLOW_ARM_SELF_COLLISION,ARM_SELF_CLEARANCE


class CommandGeometryCache:
    def __init__(self,safety,capacity=32):
        self.safety=safety;self.capacity=capacity;self.cache=OrderedDict();self.hits=0;self.queries=0

    def state(self,q):
        key=np.asarray(q,dtype=np.float64).tobytes()
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return self.cache[key]
        self.queries+=1;s=self.safety;s.collision_reason(q)
        state=(s.arm_clearance_state(),s._base_reason,s.clearance_state()[0])
        self.cache[key]=state
        if len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return state

    def command_reason(self,measured_q,command):
        proposed_arm,proposed_base,proposed_clearance=self.state(command)
        measured_arm,measured_base,measured_clearance=self.state(measured_q)
        if any(d<=0. for d in measured_arm.values()) or any(d<=0. for d in proposed_arm.values()):
            return SHALLOW_ARM_SELF_COLLISION
        if not self.safety._arm_exit_allowed(measured_arm,proposed_arm):return ARM_SELF_CLEARANCE
        if proposed_base=='NON_TIP_CLEARANCE':
            if measured_base==proposed_base and proposed_clearance>0. and proposed_clearance>=measured_clearance-1e-6:
                return None
        return proposed_base
