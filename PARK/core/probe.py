"""Four point-contact KDEs and the local ball-mass field in the measured palm frame."""
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import ncx2

from .common import FINGERS, ROOT, digest, read, table, write


def log_density(query, centers, weights, h):
    const = -3 * np.log(np.sqrt(2 * np.pi) * h)
    return np.concatenate([const + logsumexp(np.log(weights) - cdist(q, centers, "sqeuclidean") / (2*h*h),
                                             axis=1) for q in np.array_split(query, 1 + len(query)//128)])


def direct_mass(query, centers, weights, h, radius):
    d2 = cdist(query, centers, "sqeuclidean") / h**2
    near = d2 < (radius/h + 9)**2
    prob = np.zeros_like(d2)
    prob[near] = ncx2.cdf((radius/h)**2, 3, d2[near])
    return prob @ weights


class BallField:
    def __init__(self, centers, weights, h, radius, spacing, origin, grid):
        self.centers, self.weights = centers, weights
        self.h, self.radius, self.spacing = float(h), float(radius), float(spacing)
        self.origin, self.grid = origin, grid

    @classmethod
    def fit(cls, centers, weights, h, radius, spacing):
        # Discretize the mixture locations linearly. The convolution kernel is the
        # integrated 3-D Gaussian for each displacement, including the center voxel.
        pad = int(np.ceil((radius + 7*h) / spacing)) + 1
        origin = (np.floor(centers.min(0)/spacing) - pad) * spacing
        position = (centers - origin) / spacing
        bins = np.floor(position).astype(int)
        histogram = np.zeros(bins.max(0) + pad + 3)
        for corner in np.ndindex(2, 2, 2):
            fraction = np.prod(np.where(corner, position - bins, 1 - (position - bins)), axis=1)
            np.add.at(histogram, tuple((bins + corner).T), fraction * weights)
        axis = np.arange(-pad, pad+1)**2
        d2 = axis[:, None, None] + axis[None, :, None] + axis[None, None, :]
        lookup = ncx2.cdf((radius/h)**2, 3, np.arange(d2.max()+1) * (spacing/h)**2)
        field = fftconvolve(histogram, lookup[d2], mode="same")
        if not np.isfinite(field).all() or field.min() < -1e-9 or field.max() > 1 + 1e-9:
            raise ValueError("Invalid integrated probability field")
        return cls(centers, weights, h, radius, spacing, origin, field.clip(0, 1).astype("float32"))

    def mass(self, xyz):
        indices = (xyz.reshape(-1, 3) - self.origin) / self.spacing
        return map_coordinates(self.grid, indices.T, order=1, prefilter=False,
                               mode="constant", cval=0.).reshape(xyz.shape[:-1])

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**dict(data))


class Probe:
    def __init__(self, models, variant):
        if variant != "D":
            raise ValueError("Unsupported PARK probe configuration")
        self.fields = [BallField.load(models / (name + ".npz")) for name in FINGERS]

    def fingers(self, points):
        return np.stack([field.mass(points) for field in self.fields], axis=-1)



def contacts(directory, cfg):
    from Module.palmreach6.__main__ import FIXTURES
    index = read(directory / "index.json")
    checkpoint = read(FIXTURES / "fixtures.json")["checkpoint"]
    if index["version"] != 1 or index["checkpoint"] != checkpoint or tuple(index["fingers"]) != FINGERS:
        raise ValueError("Historical fingertip schema or checkpoint mismatch")
    rows, episodes = [], []
    objects = {}
    for record in index["records"]:
        file = directory / record["file"]
        if digest(file) != record["sha256"]:
            raise ValueError(f"Corrupted historical episode {file}")
        split, obj = record["group"], record["source"]["object"]
        if objects.setdefault(obj, split) != split or obj in read(FIXTURES / "fixtures.json")["objects"]:
            raise ValueError(f"Object leakage: {obj}")
        with np.load(file, allow_pickle=False) as data:
            runs = []
            for i, name in enumerate(FINGERS):
                mask = (data["finger"] == i) & np.isfinite(data["palm"]).all(1)
                p = data["palm"][mask]
                ticks = data["tick"][mask]
                if len(p) and (np.diff(ticks) <= 0).any():
                    raise ValueError(f"Duplicated or unordered fresh contacts in {file}")
                # Unique palm voxels within this finger and episode. Dwell time
                # cannot increase a location's weight, even after a later revisit.
                _, first = np.unique(np.floor(p/cfg["dedup_m"]).astype(int), axis=0, return_index=True)
                p = p[np.sort(first)]
                limit = cfg["max_per_episode"] if split == "train" else cfg["validation_per_episode"]
                chosen = np.linspace(0, len(p)-1, min(len(p), limit), dtype=int)
                runs.append(p[chosen])
                rows.append(dict(episode=record["name"], object=obj, split=split, finger=name,
                                 episode_raw_valid=record["source"]["raw_valid"], valid_50ms=int(mask.sum()),
                                 unique_palm_voxels=len(p), used=len(chosen), archive_sha256=record["sha256"]))
            episodes.append((split, runs))
    return rows, episodes


def balanced(episodes, split, finger):
    groups = [p[finger] for role, p in episodes if role == split and len(p[finger])]
    if not groups:
        raise ValueError(f"No valid {FINGERS[finger]} contacts in {split}")
    return np.vstack(groups), np.concatenate([np.full(len(g), 1/(len(groups)*len(g))) for g in groups])


def fit(directory, models, cfg):
    from shutil import copyfile
    rows, episodes = contacts(directory, cfg)
    models.mkdir(parents=True, exist_ok=True)
    table(models / "episodes.csv", rows)
    rng = np.random.default_rng(cfg["seed"])
    details = {}
    for i, name in enumerate(FINGERS):
        centers, weight = balanced(episodes, "train", i)
        val, vw = balanced(episodes, "validation", i)
        test, tw = balanced(episodes, "test", i)
        tree = cKDTree(centers)
        draws = centers[rng.choice(len(centers), 4000, p=weight)]
        noise = rng.normal(size=draws.shape)
        sweep = []
        for h in cfg["bandwidths_m"]:
            leak = (tree.query(draws + h*noise)[0] > .008).mean()
            sweep.append(dict(h=h, validation_log_density=float(vw @ log_density(val, centers, weight, h)),
                              mass_beyond_8mm_from_support=float(leak)))
        eligible = [s for s in sweep if s["mass_beyond_8mm_from_support"] <= .05]
        if not eligible:
            raise ValueError(f"No bandwidth preserves empirical support for {name}")
        h = max(eligible, key=lambda s: s["validation_log_density"])["h"]
        radius, spacing = cfg["radius_m"], cfg["grid_m"]
        field = BallField.fit(centers, weight, h, radius, spacing)
        np.savez_compressed(models / (name + ".npz"), **vars(field))
        query = draws[:160] + rng.normal(0, radius, (160, 3))
        prob = field.mass(query)
        exact = direct_mass(query, centers, weight, h, radius)
        observed = (cdist(query, test) <= radius) @ tw
        error = float(np.max(abs(exact - prob)))
        if error > .025 or not np.all((prob >= 0) & (prob <= 1)):
            raise ValueError(f"Ball integral check failed for {name}: {error}")
        details[name] = dict(bandwidth_m=h, centers=len(centers), selection=sweep,
                             max_integral_error=error, test_mass_mae=float(abs(prob-observed).mean()),
                             test_log_density=float(tw @ log_density(test, centers, weight, h)),
                             calibration=[dict(predicted=float(prob[ix].mean()), observed=float(observed[ix].mean()),
                                               count=len(ix)) for ix in np.array_split(np.argsort(prob), 5)])
    source = ROOT / "outputs/PARK/models/probe-inputs/fk.npz"
    copyfile(source, models / "fk.npz")
    write(models / "probe.json", dict(version=1, frame="measured palm XYZ metres", config=cfg,
          historical_index_sha256=digest(directory / "index.json"), fingers=details, fk_source_sha256=digest(source)))
