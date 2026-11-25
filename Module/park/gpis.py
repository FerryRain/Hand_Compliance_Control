"""Bounded GPyTorch posterior with fixed calibration and exact CUDA marginal solves."""
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import gpytorch
import numpy as np
from scipy.spatial import cKDTree
import torch

from .settings import unit, write_json


@dataclass(frozen=True)
class Calibration:
    kernel: str = "matern32"
    length_m: float = .028
    sites: int = 96
    offset_m: float = .005
    output_scale_m: float = .025
    prior_m: float = .009
    noise: float = .008


def spread(points, limit):
    if len(points) <= limit:
        return np.arange(len(points))
    idx = np.empty(limit, dtype=int)
    d = np.full(len(points), np.inf)
    row = np.argmin(np.linalg.norm(points - points.mean(0), axis=1))
    for j in range(limit):
        idx[j] = row
        diff = points - points[row]
        np.minimum(d, np.einsum("ij,ij->i", diff, diff), out=d)
        d[row] = -1
        row = np.argmax(d)
    return idx


class SignedGP(gpytorch.models.ExactGP):
    def __init__(self, cfg):
        super().__init__(None, None, gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = (gpytorch.kernels.MaternKernel(nu=1.5) if cfg.kernel == "matern32"
                             else gpytorch.kernels.RBFKernel())
        self.covar_module.lengthscale = cfg.length_m
        self.mean_module.constant = cfg.prior_m / cfg.output_scale_m
        self.likelihood.noise = cfg.noise
        self.requires_grad_(False)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class GPIS:
    def __init__(self, cfg, state=None):
        if not torch.cuda.is_available():
            raise RuntimeError("PARK requires CUDA for GPIS conditioning and inference")
        self.cfg = cfg
        self.gp = SignedGP(cfg).cuda().eval()
        if state is not None:
            self.gp.load_state_dict(state)
        self.origin = None
        self.factor = None

    @classmethod
    def load(cls, path):
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record["version"] != 1:
            raise ValueError("Unsupported GPIS checkpoint")
        return cls(Calibration(**record["calibration"]), record["state"])

    def save(self, path):
        torch.save(dict(version=1, calibration=asdict(self.cfg),
                        state={k: v.cpu() for k, v in self.gp.state_dict().items()}), path)

    @torch.no_grad()
    def condition(self, points, normals):
        cfg = self.cfg
        ids = spread(points, cfg.sites)
        p, n = points[ids], normals[ids]
        if not len(p):
            raise ValueError("A GPIS needs measured surface contacts")
        oriented = np.linalg.norm(n, axis=1) > .99
        delta = cfg.offset_m * unit(n[oriented])
        x = np.vstack((p, p[oriented] + delta, p[oriented] - delta))
        self.origin = p.mean(0)
        x = torch.as_tensor(x - self.origin, device="cuda", dtype=torch.float32)
        y = torch.as_tensor(np.repeat([0., cfg.offset_m, -cfg.offset_m], [len(p), len(delta), len(delta)]),
                            device="cuda", dtype=torch.float32) / cfg.output_scale_m
        self.gp.set_train_data(x, y, strict=False)
        k = self.gp.covar_module(x).to_dense()
        k.diagonal().add_(self.gp.likelihood.noise.reshape(()))
        self.factor = torch.linalg.cholesky(k)
        self.alpha = torch.cholesky_solve((y - self.gp.mean_module(x))[:, None], self.factor)
        torch.cuda.synchronize()

    @torch.no_grad()
    def query(self, points):
        if self.factor is None:
            raise ValueError("Condition the GPIS before querying")
        x = torch.as_tensor(np.asarray(points) - self.origin, device="cuda", dtype=torch.float32)
        train = self.gp.train_inputs[0]
        cross = self.gp.covar_module(x, train).to_dense()
        white = torch.linalg.solve_triangular(self.factor, cross.T.contiguous(), upper=False)
        mean = (self.gp.mean_module(x) + (cross @ self.alpha)[:, 0]) * self.cfg.output_scale_m
        variance = (1 - white.square().sum(0)).clamp(0, 1) * self.cfg.output_scale_m**2
        diff = train[None, :, :] - x[:, None, :]
        length = self.gp.covar_module.lengthscale
        if self.cfg.kernel == "matern32":
            rate = np.sqrt(3.) / length
            derivative = rate.square() * torch.exp(-rate * torch.linalg.vector_norm(diff, dim=2))
        else:
            derivative = cross / length.square()
        grad = ((derivative * self.alpha.T)[:, :, None] * diff).sum(1) * self.cfg.output_scale_m
        return tuple(v.cpu().numpy() for v in (mean, variance, grad))


def calibration_cases(episodes):
    for ep in episodes:
        _, ids = np.unique(np.floor(ep.world / .0045).astype(int), axis=0, return_index=True)
        ids = np.sort(ids)
        rng = np.random.default_rng(742)
        ids = rng.permutation(ids)
        # Disjoint physical voxels even when the recorded hand dwells on a contact.
        cut = min(300, len(ids) // 2)
        train, test = ids[:cut], ids[cut:]
        near = cKDTree(ep.world[train]).query(ep.world[test])[0] <= .038
        test = test[near][:220]
        if len(test) >= 8:
            yield ep.world[train], ep.normals[train], ep.world[test], ep.normals[test]


def benchmark(cfg, cases):
    gp = GPIS(cfg)
    errors, angles, signed_errors, accuracy, latency, update = [], [], [], [], [], []
    for train, normal, test, truth_normal in cases:
        t = perf_counter()
        gp.condition(train, normal)
        update.append((perf_counter() - t) * 1000)
        probe = np.vstack((test, test + cfg.offset_m * truth_normal, test - cfg.offset_m * truth_normal))
        for _ in range(3):
            t = perf_counter()
            values, _, grad = gp.query(probe)
            latency.append((perf_counter() - t) * 1000)
        zero, outside, inside = np.split(values, 3)
        g = grad[:len(test)]
        errors.extend(abs(zero) / np.maximum(np.linalg.norm(g, axis=1), .20))
        angles.extend(np.rad2deg(np.arccos(np.clip(np.sum(unit(g) * truth_normal, axis=1), -1., 1.))))
        signed_errors.extend(np.r_[abs(outside - cfg.offset_m), abs(inside + cfg.offset_m)])
        accuracy.extend(np.r_[outside > 0, inside < 0])
    if not errors:
        raise ValueError("No spatially independent calibration samples")
    return dict(surface_mae_mm=float(np.mean(errors) * 1000), signed_mae_mm=float(np.mean(signed_errors) * 1000),
                sign_accuracy=float(np.mean(accuracy)), normal_median_deg=float(np.median(angles)),
                query_p50_ms=float(np.median(latency)), update_p50_ms=float(np.median(update)),
                heldout_voxels=len(errors), queries_per_call=[3 * len(c[2]) for c in cases])


def calibrate(groups, directory):
    validation = list(calibration_cases(groups["validation"]))
    test = list(calibration_cases(groups["test"]))
    GPIS(Calibration()).condition(validation[0][0][:24], validation[0][1][:24])
    rows = []
    for kernel in ("rbf", "matern32"):
        for length in (.022, .036):
            for sites in (48, 96, 192):
                cfg = Calibration(kernel, length, sites)
                rows.append(dict(config=asdict(cfg), **benchmark(cfg, validation)))
    # Predeclared tolerance for the speed / accuracy trade-off. Test objects never select parameters.
    best = min(r["surface_mae_mm"] for r in rows)
    eligible = [r for r in rows if r["surface_mae_mm"] <= best + .20]
    chosen = min(eligible, key=lambda r: r["update_p50_ms"] + r["query_p50_ms"])
    cfg = Calibration(**chosen["config"])
    GPIS(cfg).save(Path(directory) / "gpis.pt")
    write_json(Path(directory) / "gpis-calibration.json",
               dict(selection="fastest median update + query within 0.20 mm of best validation surface MAE",
                    gpu=torch.cuda.get_device_name(), gpytorch=gpytorch.__version__, torch=torch.__version__,
                    validation=rows, selected=asdict(cfg), test=benchmark(cfg, test)))

