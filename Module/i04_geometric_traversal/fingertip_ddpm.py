"""Cache the released epsilon DDPM schedule without changing the policy or noise draws."""
import torch


class CachedDDPM:
    def __init__(self, scheduler, steps, device, *, conditioning, predict_noise):
        config = scheduler.config
        if (config.prediction_type != 'epsilon' or config.variance_type != 'fixed_small'
                or config.thresholding):
            raise ValueError('Cached DDPM requires the released epsilon/fixed-small schedule')
        self.clip = config.clip_sample_range if config.clip_sample else None
        self.conditioning = conditioning
        self.predict = predict_noise
        self.steps = steps
        scheduler.set_timesteps(steps, device='cpu')
        indices = scheduler.timesteps.tolist()
        alphas = scheduler.alphas_cumprod.cpu()
        tensors = scheduler.timesteps.to(device).unbind()
        self.schedule = []
        for timestep, next_step, device_t in zip(indices, indices[1:] + [-1], tensors):
            alpha = alphas[timestep]
            next_alpha = alphas[next_step] if next_step >= 0 else scheduler.one.cpu()
            retained = alpha / next_alpha
            beta, noisy = 1 - retained, 1 - alpha
            clean_weight = next_alpha.sqrt() * beta / noisy
            current_weight = retained.sqrt() * (1 - next_alpha) / noisy
            sigma = ((1 - next_alpha) / noisy * beta).clamp_min(1e-20).sqrt()
            self.schedule.append((device_t, noisy.sqrt().item(), alpha.sqrt().item(),
                                  clean_weight.item(), current_weight.item(), sigma.item() if timestep > 0 else 0.))

    def __call__(self, models, arm, x, generator, *, thumb_stream=None, denoise_graph=None, denoise_steps=10):
        if denoise_steps != self.steps:
            raise ValueError('Requested DDPM steps differ from the cached schedule')
        condition = self.conditioning(models, arm, x)
        if denoise_graph is not None:
            denoise_graph.update_conditions(condition)
        value = torch.randn((len(x), 16, 16), device=x.device, generator=generator)
        for timestep, noise_weight, signal_weight, clean_weight, current_weight, sigma in self.schedule:
            if denoise_graph is not None:
                epsilon = denoise_graph(value, timestep)
            else:
                epsilon = self.predict(models, arm, value, timestep, condition, thumb_stream=thumb_stream)
            clean = (value - noise_weight * epsilon) / signal_weight
            if self.clip is not None:
                clean = clean.clamp(-self.clip, self.clip)
            value = clean_weight * clean + current_weight * value
            if sigma:
                noise = torch.randn(value.shape, generator=generator, device=value.device, dtype=value.dtype)
                value = value + sigma * noise
        return value
