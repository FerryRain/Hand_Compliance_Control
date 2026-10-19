"""Independent relief candidate: scale a whole finger step before final guards.

The original force threshold, measured normal, site Jacobian, damping, velocity
and applied-reference integration are unchanged. A scalar in [0,1] fits that
same direction inside the existing slew/joint box. This is not a guarantee of
force reduction; native dynamics and all caller safety guards remain required.
"""
from __future__ import annotations
import numpy as np
from Module.e05_physics.scene import FINGERS
from .reference_force_release import applied_reference_force_release


def scale_step_to_joint_box(previous, delta, lower, upper, maximum_step=.0024):
    previous, delta, lower, upper = [np.asarray(x,dtype=float) for x in (previous,delta,lower,upper)]
    if (previous.shape!=(4,) or any(x.shape!=(4,) for x in (delta,lower,upper))
            or not all(np.isfinite(x).all() for x in (previous,delta,lower,upper))
            or np.any(lower>=upper) or not np.isfinite(maximum_step)
            or not 0.<maximum_step<=.0024):
        raise ValueError('Finite four-joint data and an original-or-tighter slew bound required')
    if np.any(previous<lower) or np.any(previous>upper):
        return np.zeros(4),dict(feasible=False,alpha=0.,reason='PRIOR_REFERENCE_OUTSIDE_JOINT_BOX')
    lo=np.maximum(-maximum_step,lower-previous)
    hi=np.minimum(maximum_step,upper-previous)
    ratios=[1.]
    for i,value in enumerate(delta):
        if value>0.:ratios.append(float(hi[i]/value))
        elif value<0.:ratios.append(float(lo[i]/value))
    alpha=float(np.clip(min(ratios),0.,1.))
    # Round down the scalar rather than independently repairing components.
    if 0.<alpha<1.:alpha=float(np.nextafter(alpha,0.))
    step=alpha*delta
    if (np.any(previous+step<lower) or np.any(previous+step>upper)
            or np.any(abs(step)>maximum_step)):
        return np.zeros(4),dict(feasible=False,alpha=0.,reason='FLOATING_POINT_BOX_REJECTION')
    return step,dict(feasible=True,alpha=alpha,
        reason='DIRECTION_BLOCKED_HOLD' if alpha==0. and np.any(delta) else 'DIRECTION_PRESERVED')


def direction_preserving_reference_force_release(handles,data,command,previous_applied,
        forces,normals,config,*,lower,upper,maximum_joint_step=.0024):
    """Same raw DLS proposal, then one common scalar per actually loaded finger."""
    raw=applied_reference_force_release(handles,data,command,previous_applied,forces,normals,config)
    previous=np.asarray(previous_applied,dtype=float)
    lower,upper=np.asarray(lower,dtype=float),np.asarray(upper,dtype=float)
    if lower.shape!=(23,) or upper.shape!=(23,):
        raise ValueError('Expected complete canonical joint bounds')
    result=raw.copy();rows=[]
    for finger in range(4):
        if forces[finger]<4. or np.linalg.norm(normals[finger])<.5:
            rows.append(dict(loaded=False));continue
        columns=np.asarray([int(n) for n in FINGERS[finger].joint_names])+7
        delta=raw[columns]-previous[columns]
        step,diag=scale_step_to_joint_box(previous[columns],delta,lower[columns],upper[columns],maximum_joint_step)
        result[columns]=previous[columns]+step
        rows.append(dict(loaded=True,raw_delta_rad=delta.tolist(),applied_delta_rad=step.tolist(),**diag))
    return result,dict(fingers=rows,policy='COMMON_SCALAR_PER_LOADED_FINGER_OR_EXPLICIT_HOLD',
                       no_force_reduction_claim=True)
