"""Fixed-hyperparameter GPyTorch signed-distance posterior on bounded tactile support."""
from time import perf_counter

import gpytorch
import numpy as np
import torch

from .common import ROOT, digest, dispersed, read, unit, write


class Belief:
    def __init__(self, config, state=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the PARK-3 posterior")
        self.cfg = config
        kernels = {"rbf": gpytorch.kernels.RBFKernel, "matern32": lambda: gpytorch.kernels.MaternKernel(nu=1.5)}
        self.kernel = kernels[config["kernel"]]().cuda().eval()
        self.kernel.lengthscale = config["length_m"]
        if state:
            self.kernel.load_state_dict(state)
        self.kernel.requires_grad_(False)
        self.center = self.sites = self.factor = None
        self.support = np.empty((0, 6))
        self.scale = config["output_scale_m"]

    @classmethod
    def load(cls, path):
        rec = torch.load(path, map_location="cpu", weights_only=True)
        if rec["version"] != 1:
            raise ValueError("Unknown GPIS checkpoint")
        return cls(rec["config"], rec["kernel_state"])

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device="cuda")

    def covariance(self, a, b):
        return self.kernel(a, b).to_dense()

    @torch.inference_mode()
    def condition(self, points, normals):
        if not len(points):
            return "empty"
        ix = np.sort(dispersed(points, self.cfg["sites"]))
        p, n = points[ix], unit(normals[ix])
        support = np.column_stack((p, n))
        previous = len(self.support)
        if np.array_equal(support, self.support):
            return "cached"
        if self.center is None:
            self.center = p[0].copy()
        offset = self.cfg["offset_m"]
        xyz = np.stack((p, p + offset*n, p - offset*n), axis=1).reshape(-1, 3)
        x = self.tensor(xyz - self.center)
        incremental = previous > 0 and previous < len(p) and np.array_equal(support[:previous], self.support)
        if incremental:
            block = x[3*previous:]
            cross = self.covariance(self.sites, block)
            v = torch.linalg.solve_triangular(self.factor, cross, upper=False)
            schur = self.covariance(block, block) - v.T @ v
            schur.diagonal().add_(self.cfg["noise"])
            new_factor = torch.zeros((len(x), len(x)), device=x.device)
            k = len(self.sites)
            new_factor[:k, :k] = self.factor
            new_factor[k:, :k] = v.T
            new_factor[k:, k:] = torch.linalg.cholesky(schur)
            self.factor = new_factor
        else:
            gram = self.covariance(x, x)
            gram.diagonal().add_(self.cfg["noise"])
            self.factor = torch.linalg.cholesky(gram)
        y = self.tensor(np.tile([0, offset, -offset], len(p)) - self.cfg["prior_m"])
        self.alpha = torch.cholesky_solve(y[:, None], self.factor)
        self.sites, self.support = x, support
        torch.cuda.synchronize()
        return "append" if incremental else "rebuild"

    @torch.inference_mode()
    def query(self, xyz):
        if self.sites is None:
            raise ValueError("No tactile support for the GPIS")
        x = self.tensor(np.asarray(xyz) - self.center)
        k = self.covariance(x, self.sites)
        mean = self.cfg["prior_m"] + (k @ self.alpha).flatten()
        whitened = torch.linalg.solve_triangular(self.factor, k.T.contiguous(), upper=False)
        variance = (1 - whitened.square().sum(0)).clamp(0, 1) * self.scale**2
        delta = self.sites[None] - x[:, None]
        h = self.kernel.lengthscale
        if self.cfg["kernel"] == "rbf":
            deriv = k / h.square()
        else:
            rate = np.sqrt(3.)/h
            deriv = rate.square() * torch.exp(-rate * delta.norm(dim=-1))
        gradient = torch.einsum("qs,qsd->qd", deriv * self.alpha.T, delta)
        return tuple(v.cpu().numpy() for v in (mean, variance, gradient))


def prepare(directory, models, cfg):
    from Module.park.history import load_episodes
    from Module.park.gpis import calibration_cases
    source = ROOT / cfg["calibration_file"]
    calibrated = read(source)
    rows = calibrated["validation"]
    cutoff = min(r["surface_mae_mm"] for r in rows) + cfg["validation_tolerance_mm"]
    selected = min((r for r in rows if r["surface_mae_mm"] <= cutoff),
                   key=lambda r: r["query_p50_ms"] + r["update_p50_ms"])["config"]
    model = Belief(selected)
    record = dict(version=1, config=selected, kernel_state={k: v.cpu() for k, v in model.kernel.state_dict().items()})
    torch.save(record, models / "gpis.pt")
    episodes = list(load_episodes(directory, read(directory / "index.json")["checkpoint"]))
    cases = list(calibration_cases([e for e in episodes if e.group == "test"]))
    # Warm CUDA before the measured held-out posterior timings.
    model.condition(cases[0][0][:16], cases[0][1][:16])
    model.query(cases[0][2][:8])
    torch.cuda.reset_peak_memory_stats()
    update, query, errors, angles, signs = [], [], [], [], []
    for points, normals, heldout, truth in cases:
        t = perf_counter()
        model.condition(points, normals)
        update.append(1000*(perf_counter()-t))
        queries = np.vstack((heldout, heldout+selected["offset_m"]*truth, heldout-selected["offset_m"]*truth))
        for _ in range(5):
            t = perf_counter()
            m, _, grad = model.query(queries)
            query.append(1000*(perf_counter()-t))
        zero, out, inside = np.split(m, 3)
        g = grad[:len(zero)]
        errors.extend(np.abs(zero) / np.maximum(np.linalg.norm(g, axis=1), .2))
        angles.extend(np.rad2deg(np.arccos((unit(g)*truth).sum(1).clip(-1, 1))))
        signs.extend(np.r_[out > 0, inside < 0])
    write(models / "gpis.json", dict(config=selected, calibration_sha256=digest(source), validation_candidates=rows,
          gpu=torch.cuda.get_device_name(), torch=torch.__version__, gpytorch=gpytorch.__version__,
          split="ovoid, disjoint world voxels", heldout=len(errors), surface_mae_mm=1000*float(np.mean(errors)),
          normal_median_deg=float(np.median(angles)), sign_accuracy=float(np.mean(signs)),
          update_ms=update, query_ms=query, queries_per_call=[3*len(c[2]) for c in cases],
          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
