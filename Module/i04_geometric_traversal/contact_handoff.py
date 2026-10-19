"""Preserve already applied servo preload when reseeding measured-contact MCC.

The stored hand-reference bias is ``last_applied - new_nominal`` only for
fingers with actual safe contact at the handoff. It is a command reference,
not a pose observation: FK of a preloaded servo command need not equal the
physical pose. The unchanged GraceMCC then adds its usual measured-force
correction. This class never writes MuJoCo state, awards coverage, bypasses
command geometry checks, or raises any original MCC/plant limit.

The caller remains responsible for force relief, native integration, command
slew/joint limits and collision checks on the *complete returned command*.
In particular, existing command-guard handling of pad compression is distinct
from static whole-path collision checks on uncompressed geometric poses.
"""
from __future__ import annotations

import numpy as np

from Module.e05_physics.scene import FINGERS
from .surface_slide import GraceMCC


class ContactPreservingMCC(GraceMCC):
    """Unchanged GraceMCC plus an explicitly seeded, constant hand bias.

    ``seed_handoff`` replaces the old bias and resets old MCC history; repeated
    handoffs therefore cannot add the old correction twice. Pass the actual
    last applied canonical 23-joint command, after the caller's final guards.
    No arm offset is retained. An intentional-air reset discards all preload
    and grace history. A temporary contact loss alone does not manufacture a
    new bias or claim contact; the original 100 ms GraceMCC rule still applies.
    """
    contact_min_n = .2
    safe_force_max_n = 8.

    def __init__(self, handles, dt_s, nominal_q=None, last_applied_q=None,
                 active=None, forces_n=None, *, intentional_air=False):
        if not np.isfinite(dt_s) or dt_s <= 0.:
            raise ValueError('Expected a finite positive native timestep')
        super().__init__(handles, dt_s)
        self._held_bias_rad = np.zeros(16)
        self._seeded_fingers = np.zeros(4, dtype=bool)
        supplied = [x is not None for x in (nominal_q, last_applied_q, active, forces_n)]
        if any(supplied):
            if not all(supplied):
                raise ValueError('A handoff requires nominal, applied command, active and forces')
            self.seed_handoff(nominal_q, last_applied_q, active, forces_n,
                              intentional_air=intentional_air)

    @property
    def held_bias_rad(self):
        """A copy of the 16 canonical hand-joint reference offsets."""
        return self._held_bias_rad.copy()

    @property
    def seeded_fingers(self):
        """Actual safe contact mask used at the last handoff, not current contact."""
        return self._seeded_fingers.copy()

    def reset(self):
        """Discard reference preload and all prior compliance/grace history."""
        GraceMCC.__init__(self, self.handles, self.dt)
        self._held_bias_rad = np.zeros(16)
        self._seeded_fingers = np.zeros(4, dtype=bool)

    def seed_handoff(self, nominal_q, last_applied_q, active, forces_n, *,
                     intentional_air=False):
        nominal = np.asarray(nominal_q, dtype=float)
        previous = np.asarray(last_applied_q, dtype=float)
        measured = np.asarray(active, dtype=bool)
        force = np.asarray(forces_n, dtype=float)
        if (nominal.shape != (23,) or previous.shape != (23,)
                or measured.shape != (4,) or force.shape != (4,)
                or not all(np.isfinite(x).all() for x in (nominal, previous, force))
                or np.any(force < 0.)):
            raise ValueError('Expected finite canonical 23-joint references and four-finger measurements')
        self.reset()
        safe = measured & (force >= self.contact_min_n) & (force < self.safe_force_max_n)
        if not intentional_air:
            self._seeded_fingers[:] = safe
            for finger in np.flatnonzero(safe):
                columns = np.asarray([int(name) for name in FINGERS[finger].joint_names])
                self._held_bias_rad[columns] = previous[columns+7] - nominal[columns+7]
        return dict(held_bias_rad=self.held_bias_rad.tolist(),
                    seeded_fingers=self.seeded_fingers.tolist(),
                    intentional_air=bool(intentional_air),
                    authority='APPLIED_SERVO_REFERENCE_ONLY_NOT_MEASURED_POSE',
                    measured_contact_claim=False, coverage_credit=False)

    def command(self, data, nominal_q, forces, normals, active, desired_force, *,
                intentional_air=False):
        nominal = np.asarray(nominal_q, dtype=float)
        if nominal.shape != (23,) or not np.isfinite(nominal).all():
            raise ValueError('Expected a finite canonical 23-joint nominal reference')
        if intentional_air:
            self.reset()
            return nominal.copy()
        reference = nominal.copy()
        reference[7:] += self._held_bias_rad
        return super().command(data, reference, forces, normals, active, desired_force)
