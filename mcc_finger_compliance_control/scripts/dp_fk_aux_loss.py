"""Noise-prediction diffusion loss with an optional fingertip FK term."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_hand_fk import TorchHandFK


def diffusion_with_fk_loss(
    policy,
    batch: dict[str, Tensor],
    *,
    fk: TorchHandFK | None,
    action_mean: Tensor,
    action_scale: Tensor,
    fk_weight: float,
    fk_max_noise_timestep: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return total, epsilon MSE, and fingertip MSE in square millimetres.

    The denoising path matches LeRobot's ``compute_loss`` for epsilon
    prediction.  FK uses the denormalized clean-action estimate at only the
    configured low-noise timesteps, not the noisy action itself.
    """
    diffusion = policy.diffusion
    if diffusion.config.prediction_type != "epsilon":
        raise ValueError("FK auxiliary loss currently supports epsilon prediction only")
    if fk_weight < 0.0:
        raise ValueError("fk_weight must be nonnegative")
    trajectory = batch["action"]
    if trajectory.shape[-1] != 16:
        raise ValueError("FK auxiliary loss requires 16-D joint actions")
    if action_mean.shape != (16,) or action_scale.shape != (16,):
        raise ValueError("action normalization must have shape (16,)")
    train_steps = int(diffusion.noise_scheduler.config.num_train_timesteps)
    if not 0 <= fk_max_noise_timestep < train_steps:
        raise ValueError("fk_max_noise_timestep is outside the training schedule")

    global_condition = diffusion._prepare_global_conditioning(batch)
    epsilon = torch.randn(trajectory.shape, device=trajectory.device)
    timestep = torch.randint(
        0, train_steps, (trajectory.shape[0],), device=trajectory.device
    ).long()
    noisy = diffusion.noise_scheduler.add_noise(trajectory, epsilon, timestep)
    predicted_epsilon = diffusion.unet(noisy, timestep, global_cond=global_condition)
    noise_error = F.mse_loss(predicted_epsilon, epsilon, reduction="none")
    if diffusion.config.do_mask_loss_for_padding:
        noise_error = noise_error * (~batch["action_is_pad"]).unsqueeze(-1)
    noise_loss = noise_error.mean()
    if fk_weight == 0.0:
        return noise_loss, noise_loss, noise_loss.new_zeros(())
    if fk is None:
        raise ValueError("TorchHandFK is required when fk_weight is positive")

    chosen = timestep <= fk_max_noise_timestep
    if not bool(chosen.any()):
        return noise_loss, noise_loss, noise_loss.new_zeros(())
    alpha = diffusion.noise_scheduler.alphas_cumprod.to(
        device=trajectory.device, dtype=torch.float32
    )[timestep[chosen]].view(-1, 1, 1)
    clean_estimate = (
        noisy[chosen].float()
        - torch.sqrt(1.0 - alpha) * predicted_epsilon[chosen].float()
    ) / torch.sqrt(alpha)
    mean = action_mean.float().view(1, 1, 16)
    scale = action_scale.float().view(1, 1, 16)
    predicted_q = clean_estimate * scale + mean
    target_q = trajectory[chosen].float() * scale + mean
    predicted_tip = fk(predicted_q)
    with torch.no_grad():
        target_tip = fk(target_q)
    square_mm = ((predicted_tip - target_tip) * 1000.0).square()
    valid_waypoints = (~batch["action_is_pad"][chosen]).float()
    weighted = square_mm * valid_waypoints[..., None, None]
    denominator = (valid_waypoints.sum() * 4 * 3).clamp_min(1.0)
    fk_loss_mm2 = weighted.sum() / denominator
    return noise_loss + fk_weight * fk_loss_mm2, noise_loss, fk_loss_mm2
