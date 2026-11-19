"""Collection candidate: carry joint commands, not a reprojected scalar offset.

The nominal posture is a virtual equilibrium. Under contact its restoration
acts only in the instantaneous normal nullspace. Only the new force error
changes the normal command; no old displacement is remapped by a new Jacobian.
This is a local velocity construction, not a nonlinear contact guarantee.
"""
import torch


def incremental_command(previous, nominal, normal_jac, mobility, normal_step,
                        found, lower, upper, rate, posture_gain, posture_step,
                        outward_travel, regularization):
    restore = (posture_gain * (nominal-previous)).clamp(-posture_step, posture_step)
    weighted = normal_jac * mobility
    denominator = (normal_jac * weighted).sum(-1)
    movable = denominator > 1.e-10
    inverse = weighted / denominator.clamp_min(1.e-10)[..., None]
    tangent = restore - (normal_jac*restore).sum(-1)[..., None] * inverse
    displacement = (normal_jac * (previous-nominal)).sum(-1)
    # Inward motion stops at the virtual equilibrium, not beyond it. A
    # changing contact normal may change this local projection; never force
    # an opposite step merely to re-establish the projected travel bound.
    inward_left = displacement.clamp_min(0)
    outward_left = (outward_travel-displacement).clamp_min(0)
    bounded_step = torch.maximum(torch.minimum(normal_step, outward_left), -inward_left)
    normal_delta = bounded_step[..., None] * weighted / (denominator+regularization)[..., None]
    contact_delta = torch.where(movable[..., None], tangent+normal_delta, 0.)
    delta = torch.where(found[..., None], contact_delta, restore)
    lo = (lower-previous).clamp_min(-rate)
    hi = (upper-previous).clamp_max(rate)
    # A shared nonnegative scale keeps the tangent/normal decomposition
    # intact while satisfying the original joint and command-rate box.
    available = torch.where(delta >= 0, hi, -lo).clamp_min(0)
    scale = (available/delta.abs().clamp_min(1.e-12)).amin(-1).clamp(0, 1)
    applied = delta * scale[..., None]
    command = previous+applied
    return command, dict(normal_before=(normal_jac*restore).sum(-1),
                        normal_after=(normal_jac*applied).sum(-1),
                        requested_signed_step=bounded_step.abs(),
                        scale=scale, engaged=found, movable=movable,
                        direction=torch.sign(bounded_step), enabled=found)
