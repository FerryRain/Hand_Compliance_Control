"""Exact CUDA GPyTorch posterior for the offline oriented signed surface."""

import gpytorch
import torch


class SignedExactGP(gpytorch.models.ExactGP):
  def __init__(self, train, labels, length, noise):
    options = dict(device=train.device, dtype=train.dtype)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(**options)
    likelihood.noise = torch.tensor(noise, **options)
    super().__init__(train, labels, likelihood)
    self.mean_module = gpytorch.means.ZeroMean()
    self.covar_module = gpytorch.kernels.MaternKernel(nu=1.5).to(**options)
    self.covar_module.lengthscale = length

  def forward(self, points):
    return gpytorch.distributions.MultivariateNormal(
      self.mean_module(points), self.covar_module(points)
    )


def cuda_posterior(train, labels, length, noise, device):
  """Keep one factorization resident and return just the mean for each grid batch."""
  locations = torch.as_tensor(train, device=device, dtype=torch.float64)
  targets = torch.as_tensor(labels, device=device, dtype=torch.float64)
  model = SignedExactGP(locations, targets, length, noise).eval()

  def predict(points):
    with (
      torch.inference_mode(),
      gpytorch.settings.max_cholesky_size(len(train) + 1),
      gpytorch.settings.skip_posterior_variances(True),
    ):
      query = torch.as_tensor(points, device=device, dtype=locations.dtype)
      return model(query).mean.cpu().numpy()

  return predict
