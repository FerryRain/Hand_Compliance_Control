"""Conservative broad-phase filtering for the frozen closest-pair contract.

For finite enclosing spheres, ||c1-c2||-r1-r2 is a lower bound on positive
surface separation. Only a strictly positive bound beyond the current best
(with a roundoff cushion) can reject a geom. Overlapping sphere candidates,
unbounded/invalid bounds, exact ties and missing best results are queried.
The native distance and endpoint-to-normal conversion remain unchanged.

MuJoCo defines geom_rbound around geom_xpos and uses it in its own sphere
filter: https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_collision_driver.c
Model dimensions/bounds must remain consistent, as for native broad phase.
"""
from __future__ import annotations

import mujoco
import numpy as np


_FINITE_TYPES = frozenset(int(kind) for kind in (
    mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE,
    mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_CYLINDER,
    mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_MESH))


def closest_finger_pair_pruned(h, data, finger):
    """Return exactly the original query schema; no contact/coverage authority."""
    geom = int(h.tip_geom_ids[finger])
    objects = np.asarray(h.object_geom_ids, dtype=int)
    objects = objects[h.model.geom_contype[objects] != 0]
    # Recompute from the current data/model every time: scratch FK and moving
    # objects cannot accidentally reuse a lower bound from an earlier pose.
    centers = data.geom_xpos[objects]
    center = data.geom_xpos[geom]
    radii = h.model.geom_rbound[objects]
    radius = float(h.model.geom_rbound[geom])
    separations = np.linalg.norm(centers-center, axis=1)
    # Compiled mesh vertices are float32. Their rounded coordinates can extend
    # sub-nanometres beyond the compiler's double radius. Enclose that rounding
    # conservatively before comparing bounds (also harmless for primitives).
    rounding = 4*np.finfo(np.float32).eps
    enclosed_radius = radius*(1.+rounding)+1e-12
    enclosed_radii = radii*(1.+rounding)+1e-12
    lower_bounds = separations-enclosed_radius-enclosed_radii
    valid = (np.isfinite(lower_bounds) & np.isfinite(radii) & (radii > 0.)
             & np.isfinite(radius) & (radius > 0.))
    valid &= np.isin(h.model.geom_type[objects], tuple(_FINITE_TYPES))
    valid &= int(h.model.geom_type[geom]) in _FINITE_TYPES
    # The constant dominates ordinary metre-scale roundoff. The additional
    # term is conservative at larger coordinates; equality is never pruned.
    cushion = 1e-10+64*np.finfo(float).eps*(
        np.linalg.norm(centers, axis=1)+np.linalg.norm(center)+abs(radius)+abs(radii))
    best = None
    for index, obj in enumerate(objects):
        obj = int(obj)
        if (best is not None and valid[index]
                and lower_bounds[index] > max(best['gap_m'], 0.)+cushion[index]):
            continue
        ends = np.zeros(6)
        gap = float(mujoco.mj_geomDistance(h.model, data, geom, obj, .2, ends))
        if best is None or gap < best['gap_m']:
            normal = ends[:3]-ends[3:]
            length = float(np.linalg.norm(normal))
            if length < 1e-12 or not np.isfinite(ends).all(): continue
            normal /= length
            if normal@(data.geom_xpos[geom]-ends[3:]) < 0.: normal = -normal
            best = dict(gap_m=gap, pad_point_world_m=ends[:3].copy(),
                        object_point_world_m=ends[3:].copy(), normal_world=normal,
                        object_geom_id=obj)
    return best
