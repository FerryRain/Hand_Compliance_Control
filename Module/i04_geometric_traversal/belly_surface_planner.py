"""Independent palmar-pad position AND orientation planning on the original model.

The historical point-only planners remain frozen. A geometric result never
grants contact or coverage; runtime must use individual native belly contacts.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .contracts import ExperimentConfig
from .full_surface import pad_support_vector, JointPath
from .finger_reconfiguration_planner import ellipsoid_support_derivative
from .physics import set_joint_state, pose
from .sliding_safety import SlidingSafetyPlanner
from .belly_contact import belly_ik_terms


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


class BellySurfacePlanner:
    def __init__(self, h, graph):
        self.h, self.graph = h, graph
        self.safety = SlidingSafetyPlanner(h, ExperimentConfig())
        self.scratch = mujoco.MjData(h.model)
        self.order = np.r_[h.arm_dof_adrs, h.hand_dof_adrs]

    def solve(self, root, targets, *, arm=True, clearance=.003, iterations=200,
              regularization=.0002):
        """targets maps finger -> (world surface point, object outward normal).

        Selected fingers are optimized, other fingers remain exactly fixed.
        Position uses the ellipsoid support, orientation uses the fixed +Z
        palmar axis. A vector-sum residual rejects the antiparallel solution.
        """
        h = self.h
        root = np.asarray(root,dtype=float)
        if root.shape!=(23,) or not np.isfinite(root).all() or not targets or any(i not in range(4) for i in targets):
            raise ValueError('Finite 23-joint root and at least one valid finger target required')
        if not np.isfinite(clearance) or not np.isfinite(regularization) or regularization<=0 or iterations<1:
            raise ValueError('Finite clearance, positive regularization and iteration budget required')
        columns = np.concatenate([np.arange(7) if arm else np.array([], int),
                        *[np.arange(7+4*i, 11+4*i) for i in targets]]).astype(int)
        lo, hi = self.safety.lower[columns], self.safety.upper[columns]
        weights = np.where(columns < 7, regularization*3., regularization)
        def terms(x):
            q = root.copy(); q[columns] = x
            set_joint_state(h, self.scratch, q)
            errors, jacobians, details = [], [], []
            for finger, (point, normal) in targets.items():
                geom = int(h.tip_geom_ids[finger])
                n = np.asarray(normal)/np.linalg.norm(normal)
                residual, jac, diag = belly_ik_terms(h.model,self.scratch,geom,point,n,
                    clearance_m=clearance,orientation_weight_m=.025,dof_indices=self.order[columns])
                errors.append(residual); jacobians.append(jac)
                details.append(dict(finger=finger,support_error_m=diag['position_residual_m'],
                    belly_angle_deg=diag['alignment_angle_deg'],
                    belly_patch_qualified=diag['geometry_qualified']))
            errors.append(weights*(x-root[columns]))
            jacobians.append(np.diag(weights))
            return np.concatenate(errors), np.vstack(jacobians), details
        fit = least_squares(lambda x:terms(x)[0],np.clip(root[columns],lo+1e-10,hi-1e-10),
            jac=lambda x:terms(x)[1],bounds=(lo,hi),max_nfev=iterations,
            ftol=1e-10,xtol=1e-10,gtol=1e-10)
        q = root.copy(); q[columns] = fit.x
        _, _, details = terms(fit.x)
        # Near-normal geometry is intentional: thin ellipsoid edges can be
        # outside the belly patch even with a moderately acceptable axis angle.
        accepted = all(d['support_error_m'] < .0005 and d['belly_angle_deg'] < 5.
                       and d['belly_patch_qualified'] for d in details)
        return q, dict(accepted=accepted, fingers=details, evaluations=int(fit.nfev),
                       maximum_joint_change_rad=float(np.max(abs(q-root))))

    def path(self, root, goal, *, speed=.2):
        """Dense original collision/clearance validation; no side-contact credit."""
        if not np.isfinite(speed) or speed<=0:
            raise ValueError('Finite positive joint speed required')
        count = max(2, int(np.ceil(np.max(abs(goal-root))/.004))+1)
        previous = root
        for u in np.linspace(0., 1., count)[1:]:
            trial = root+u*(goal-root)
            if not self.safety.edge_free(previous, trial):
                return None, dict(reason='STRICT_COLLISION_EDGE', fraction=float(u))
            previous = trial
        set_joint_state(self.h,self.scratch,root); start = pose(self.h,self.scratch)
        set_joint_state(self.h,self.scratch,goal); end = pose(self.h,self.scratch)
        duration=max(.5,1.875*float(np.max(abs(goal-root)))/speed)
        return JointPath(np.stack([root,goal]),np.array([duration]),start,end,0.), dict(
            reason='GEOMETRY_ONLY_PASSED', checked_states=count, duration_s=duration)
