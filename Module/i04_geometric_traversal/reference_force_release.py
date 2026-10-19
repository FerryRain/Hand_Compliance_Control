"""Integrate finger force release from the actual applied servo reference.

This changes only the reference origin of SharedContactRecovery.relieve_force.
It retains its site Jacobian, damping, measured-normal direction and configured
speed. The caller must still apply the original joint/slew/collision/native
force guards, and decide when relief starts/ends. A short loss after withdrawal
is possible; measured MCC/search handoff must be verified separately.
"""
from __future__ import annotations

import numpy as np

from Module.e05_physics.scene import FINGERS
from .physics import position_jacobian


def applied_reference_force_release(handles, data, command, previous_applied,
                                    forces, normals, config):
    """Return one relief proposal, without writing physical state or coverage.

    ``previous_applied`` must be the complete, guarded command that produced
    the current measurements. Replacing it with measured q discards the
    accumulated retreat reference and can command an inward step during relief.
    Arm commands and unloaded fingers are left to the caller's common policy.
    """
    proposed, previous = np.asarray(command, dtype=float), np.asarray(previous_applied, dtype=float)
    force, normal = np.asarray(forces, dtype=float), np.asarray(normals, dtype=float)
    if (proposed.shape != (23,) or previous.shape != (23,) or force.shape != (4,)
            or normal.shape != (4, 3)
            or not all(np.isfinite(x).all() for x in (proposed, previous, force, normal))
            or np.any(force < 0.) or not np.isfinite(config.dt_s) or config.dt_s <= 0.
            or not np.isfinite(config.force_relief_speed_m_s) or config.force_relief_speed_m_s <= 0.):
        raise ValueError('Expected finite canonical applied reference, measurements and positive relief rate')
    result = proposed.copy()
    for finger in range(4):
        if force[finger] < 4. or np.linalg.norm(normal[finger]) < .5:
            continue
        jp, _ = position_jacobian(handles, data, handles.tip_site_ids[finger])
        columns = np.asarray([int(name) for name in FINGERS[finger].joint_names])+7
        jf = jp[:, columns]
        outward = normal[finger]*config.force_relief_speed_m_s*config.dt_s
        delta = jf.T@np.linalg.solve(jf@jf.T+1e-6*np.eye(3), outward)
        result[columns] = previous[columns]+delta
    return result
