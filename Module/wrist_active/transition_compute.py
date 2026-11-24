"""Dense Matérn matrix for the small CPU patches used by the wrist follower."""

import numpy as np
import torch
import gpytorch
from scipy.spatial.distance import cdist


class DenseMatern32:
    """Supply LocalGPIS's unscaled Matérn-3/2 kernel without a lazy operator."""

    def __call__(self, first, second):
        # GPyTorch assigns the Python lengthscale through the default float
        # tensor before promoting its parameter to the training data's dtype.
        length = float(np.float32(self.lengthscale))
        distance = cdist(first.numpy(), second.numpy()) * (np.sqrt(3.) / length)
        return torch.from_numpy((1. + distance) * np.exp(-distance))


class EagerMatern32(gpytorch.kernels.MaternKernel):
    """Compute the original GPyTorch matrix directly with identical floating point operations."""
    def __init__(self):
        super().__init__(nu=1.5)
        self.double()

    def __call__(self, first, second):
        return self.forward(first,second)
