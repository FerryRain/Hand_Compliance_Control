"""FR3 self-clearance guard for sliding, without changing physical contacts.

The existing object, pad, and hand collision rules remain in FullSurfacePlanner.
Only the geometry query temporarily extends the FR3 arm margins. Actual stop
events use native contact distance and solved contact force, never query margins.
"""
from __future__ import annotations

import re

import mujoco
import numpy as np

from .full_surface import FullSurfacePlanner


ARM_SELF_CLEARANCE = 'ARM_SELF_CLEARANCE'
SHALLOW_ARM_SELF_COLLISION = 'SHALLOW_ARM_SELF_COLLISION'
_ARM_BODY = re.compile(r'^fr3v2_link\d+$')


def is_nonadjacent_arm_pair(model, body_a, body_b):
    """True only for two distinct non-neighbour FR3 rigid links.

    Weld roots matter for fixed adapters: a fixed child of one link must not
    cause that link and its direct articulated neighbour to become a new pair.
    """
    a, b = int(body_a), int(body_b)
    if a == b:
        return False
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ''
             for i in (a, b)]
    if not all(_ARM_BODY.fullmatch(name) for name in names):
        return False
    wa, wb = int(model.body_weldid[a]), int(model.body_weldid[b])
    if wa == wb:
        return False
    parent = model.body_parentid
    if int(parent[a]) == b or int(parent[b]) == a:
        return False
    parent_wa = int(model.body_weldid[int(parent[wa])])
    parent_wb = int(model.body_weldid[int(parent[wb])])
    return parent_wa != wb and parent_wb != wa


def arm_self_contacts(handles, data, force_threshold_n=.5):
    """Return measured shallow/deep arm collisions requiring a stop.

    Each item is JSON-compatible. Positive-separation margin contacts and
    forces at/below the threshold are excluded. This function neither runs
    forward dynamics nor changes the model/data supplied by the caller.
    """
    model = handles.model
    result = []
    wrench = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        ga, gb = int(contact.geom1), int(contact.geom2)
        if ga < 0 or gb < 0 or float(contact.dist) > 0.:
            continue
        ba, bb = int(model.geom_bodyid[ga]), int(model.geom_bodyid[gb])
        if not is_nonadjacent_arm_pair(model, ba, bb):
            continue
        mujoco.mj_contactForce(model, data, index, wrench)
        force = max(0., float(wrench[0]))
        if force <= force_threshold_n:
            continue
        result.append({
            'contact_index': index, 'force_n': force,
            'distance_m': float(contact.dist),
            'body_names': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                           for i in (ba, bb)],
            'geom_names': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                           for i in (ga, gb)],
        })
    return result


class SlidingSafetyPlanner(FullSurfacePlanner):
    """Preserve old geometry rules and require 1 mm between FR3 arm links.

    A measured root already inside the *positive* buffer may leave it along a
    monotonically separating path. A new pair cannot enter the buffer during
    that exit, and an actual collision never receives an exit exemption.
    """
    def __init__(self, handles, config, arm_clearance_m=.001):
        super().__init__(handles, config)
        if arm_clearance_m <= 0.:
            raise ValueError('arm_clearance_m must be positive')
        self.arm_clearance_m = float(arm_clearance_m)
        model = handles.model
        self._arm_geoms = np.array([
            g for g in range(model.ngeom)
            if _ARM_BODY.fullmatch(mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or '')
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        ], dtype=int)
        self._arm_clearances = {}
        self._base_reason = None

    def collision_reason(self, q):
        model = self.h.model
        margins = model.geom_margin.copy()
        self._arm_clearances = {}
        self._base_reason = None
        try:
            # MuJoCo uses the larger geom margin for an ordinary pair. The
            # tiny excess makes the exactly-1-mm boundary query observable.
            model.geom_margin[self._arm_geoms] = np.maximum(
                margins[self._arm_geoms], self.arm_clearance_m + 1e-6)
            self._base_reason = super().collision_reason(q)
            for contact in self.scratch.contact:
                ga, gb = int(contact.geom1), int(contact.geom2)
                if ga < 0 or gb < 0:
                    continue
                ba, bb = int(model.geom_bodyid[ga]), int(model.geom_bodyid[gb])
                if is_nonadjacent_arm_pair(model, ba, bb):
                    pair = tuple(sorted((ba, bb)))
                    self._arm_clearances[pair] = min(
                        self._arm_clearances.get(pair, float('inf')), float(contact.dist))
        finally:
            model.geom_margin[:] = margins
        if any(d <= 0. for d in self._arm_clearances.values()):
            return SHALLOW_ARM_SELF_COLLISION
        if self._base_reason:
            return self._base_reason
        if any(d < self.arm_clearance_m for d in self._arm_clearances.values()):
            return ARM_SELF_CLEARANCE
        return None

    def arm_clearance_state(self):
        """Per-body-pair minimum separation from the most recent query."""
        return self._arm_clearances.copy()

    def _arm_exit_allowed(self, previous, current):
        for pair, distance in current.items():
            if distance >= self.arm_clearance_m:
                continue
            old = previous.get(pair, float('inf'))
            if (distance <= 0. or old <= 0. or old >= self.arm_clearance_m
                    or distance < old - 1e-6):
                return False
        return True

    def edge_free(self, a, b):
        steps = max(2, int(np.ceil(np.max(abs(b-a))/.025))+1)
        self.collision_reason(a)
        previous_arm = self.arm_clearance_state()
        if any(d <= 0. for d in previous_arm.values()):
            return False
        previous_clearance, previous_depth = self.clearance_state()
        may_exit_object = self._base_reason in ('NON_TIP_CLEARANCE', 'DEEP_PAD_PENETRATION')
        for u in np.linspace(0, 1, steps):
            self.collision_reason(a+u*(b-a))
            current_arm = self.arm_clearance_state()
            if not self._arm_exit_allowed(previous_arm, current_arm):
                return False
            previous_arm = current_arm
            reason = self._base_reason
            if reason is None:
                may_exit_object = False
                continue
            clearance, depth = self.clearance_state()
            if (may_exit_object and reason in ('NON_TIP_CLEARANCE', 'DEEP_PAD_PENETRATION')
                    and clearance > 0. and clearance >= previous_clearance-1e-6
                    and depth <= previous_depth+1e-6):
                previous_clearance, previous_depth = clearance, depth
                continue
            return False
        return True

    def command_collision_reason(self, measured_q, command):
        self.collision_reason(command)
        proposed_arm = self.arm_clearance_state()
        proposed_base = self._base_reason
        proposed_clearance, _ = self.clearance_state()
        self.collision_reason(measured_q)
        measured_arm = self.arm_clearance_state()
        measured_base = self._base_reason
        measured_clearance, _ = self.clearance_state()
        if (any(d <= 0. for d in measured_arm.values())
                or any(d <= 0. for d in proposed_arm.values())):
            return SHALLOW_ARM_SELF_COLLISION
        if not self._arm_exit_allowed(measured_arm, proposed_arm):
            return ARM_SELF_CLEARANCE
        if proposed_base == 'NON_TIP_CLEARANCE':
            if (measured_base == proposed_base and proposed_clearance > 0.
                    and proposed_clearance >= measured_clearance-1e-6):
                return None
        # ARM_SELF_CLEARANCE is cleared only by the pairwise exit check above;
        # all inherited object/pad/hand reasons retain their original outcome.
        return proposed_base
