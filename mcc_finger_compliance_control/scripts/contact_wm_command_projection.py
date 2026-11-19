"""Bound a proposed joint increment without reversing a force request.

The half-space uses the measured contact Jacobian. It is a local command
consistency condition, not a prediction or guarantee of the next contact force.
The box must contain zero (holding the preceding valid command is feasible).
"""
import torch


def project_force_increment(delta, normal_jac, mobility, sign, minimum,
                            lower, upper, enabled):
    """Per-finger weighted half-space projection, then direction-preserving scale.

    A common positive scale enforces all joint/rate bounds without reversing
    the projected normal direction. A bound can reduce the requested step;
    the returned achieved projection and scale make this explicit.
    """
    a = normal_jac * sign[..., None]
    weighted = a * mobility
    denominator = (a * weighted).sum(-1)
    before = (a * delta).sum(-1)
    multiplier = (minimum - before).clamp_min(0) / denominator.clamp_min(1.e-12)
    projected = delta + multiplier[..., None] * weighted
    movable = denominator > 1.e-12
    projected = torch.where(movable[..., None], projected, torch.zeros_like(delta))
    magnitude = projected.abs().clamp_min(1.e-12)
    bounds = torch.where(projected >= 0, upper, -lower).clamp_min(0)
    scale = (bounds / magnitude).amin(-1).clamp(0, 1)
    projected = projected * scale[..., None]
    result = torch.where(enabled[..., None], projected, delta)
    return result, dict(
        normal_before=(normal_jac * delta).sum(-1),
        normal_after=(normal_jac * result).sum(-1),
        requested_signed_step=minimum,
        scale=torch.where(enabled, scale, torch.ones_like(scale)),
        engaged=enabled & ((result-delta).abs().amax(-1) > 1.e-8),
        movable=movable)
