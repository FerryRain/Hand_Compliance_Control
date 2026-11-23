"""Evaluate the information in new physical contact locations under a common GP."""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist
import trimesh

from ..acitive_02.active_gpis import NOISE_VARIANCE, VOXEL_M
from .experiment import ACTIVE2, digest, save_json
from .matched_information import CONTACT_BUDGETS, LENGTHS_M, METHODS, PEAR_SEEDS, SECONDS
from .mcc_trials import RESULTS, read
from .sweep import DT, LOAD_LIMIT
from .sweep_experiment import OUT
from .sweep_score import ref_mesh, score
from .sweep_suite import jobs


TIMES = (60, 120, 240)
EVAL_POINTS = 512
QUERY_SEED = 20261121


def measured_cells(trace):
    end = round(SECONDS / DT) + 1
    pads = trace["pad_valid"][:end]
    load = trace["region_load_n"][:end]
    force = trace["pad_force_n"][:end]
    if len(pads) != end or not np.isfinite(load).all():
        raise ValueError("Incomplete contact window or non-finite load")
    ticks, fingers = np.nonzero(pads & (load.max(1) < LOAD_LIMIT)[:, None])
    points = trace["points_m"][:end][ticks, fingers]
    if not np.isfinite(points).all() or not np.isfinite(force[ticks, fingers]).all() or np.any(force[ticks, fingers] < .2 - 1e-6):
        raise ValueError("Invalid measured touch position or force")
    _, first = np.unique(np.floor(points / VOXEL_M).astype(np.int64), axis=0, return_index=True)
    first.sort()
    return points[first], ticks[first]


def matern(left, right, length):
    scaled = cdist(left, right)
    scaled *= np.sqrt(3.) / length
    covariance = np.exp(-scaled)
    scaled += 1.
    covariance *= scaled
    return covariance


def gp_prefix(points, length, noise, queries=None, counts=()):
    """Chronological Cholesky gives every prefix without using future observations."""
    n = len(points)
    covariance = np.empty((n, n), dtype=np.float64, order="F")
    for first in range(0, n, 256):
        stop = min(n, first+256)
        covariance[:, first:stop] = matern(points[first:stop], points, length).T
    covariance.flat[::n+1] += noise
    factor = cholesky(covariance, lower=True, overwrite_a=True, check_finite=False)
    increments = np.log(factor.diagonal()) - .5*np.log(noise)
    if np.min(increments) < -1e-9:
        raise ValueError("A Gaussian information increment was negative")
    prefix = np.r_[0., np.cumsum(increments)]
    variances = {}
    if queries is not None:
        cross = matern(points, queries, length)
        solved = solve_triangular(factor, cross, lower=True, check_finite=False)
        # Average posterior point variance on the common evaluation surface.
        reduction = np.r_[0., np.cumsum(np.einsum("ij,ij->i", solved, solved) / len(queries))]
        variances = {k: float(1.-reduction[k]) for k in counts}
        if min(variances.values()) < -1e-9:
            raise ValueError("Invalid posterior variance")
    return prefix, variances


@lru_cache(maxsize=10)
def evaluation_surface(obj, start):
    mesh = trimesh.load_mesh(ref_mesh(obj, start), process=False)
    return trimesh.sample.sample_surface(mesh, EVAL_POINTS, seed=QUERY_SEED)[0]


def wrist_travel(poses):
    distance = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
    quats = poses[:, 3:7]
    dot = abs(np.einsum("ij,ij->i", quats[1:], quats[:-1]))
    angle = 2*np.arccos(np.clip(dot, 0., 1.))
    return np.r_[0., np.cumsum(distance)], np.r_[0., np.cumsum(angle)]


def evaluate(root, family, obj, start, arm, source, quality_file, seed=None):
    output = root / "evaluation" / family / obj / start
    if seed is not None:
        output = output / f"seed-{seed}"
    output = output / arm
    output.mkdir(parents=True, exist_ok=True)
    trace_path = source / "trace.npz"
    fingerprint = digest(trace_path)
    quality = read(quality_file)
    if quality["spec"]["trace_sha256"] != fingerprint:
        raise ValueError(f"Quality scores used different observations: {source}")
    spec = dict(trace_sha256=fingerprint, evaluator_sha256=digest(Path(__file__)),
                study_protocol_sha256=digest(root/"protocol.json"),
                mesh_sha256=digest(ref_mesh(obj,start)), gp_noise_variance=NOISE_VARIANCE,
                primary_length_m=LENGTHS_M[0], sensitivity_lengths_m=list(LENGTHS_M[1:]),
                query_count=EVAL_POINTS, query_seed=QUERY_SEED, observation="one zero-level sample per new cell")
    saved = output/"information.json"
    if saved.exists() and read(saved)["spec"] == spec and (output/"information-curve.npz").exists():
        return
    with np.load(trace_path, allow_pickle=False) as trace:
        points, ticks = measured_cells(trace)
        distance, angle = wrist_travel(trace["qpos"][:round(SECONDS/DT)+1])
        initial_pads = trace["pad_valid"][0]
        if not initial_pads.all():
            raise ValueError("Missing one of the four initial touches")
        initial_points = trace["points_m"][0].tolist()
    count_at = lambda t: int(np.searchsorted(ticks, t, side="right"))
    initial = count_at(0)
    if initial != 4:
        raise ValueError("Expected four distinct initial contact cells")
    indices = {f"t{s}": count_at(round(s/DT)) for s in TIMES}
    indices.update({f"n{n}": initial+n for n in CONTACT_BUDGETS if initial+n <= len(ticks)})
    for meters in (.5, 1.):
        if distance[-1] >= meters:
            tick = int(np.searchsorted(distance, meters, side="right")-1)
            indices[f"path{meters:g}"] = count_at(tick)
    # The truth mesh gets used only for the secondary surface-variance diagnostic.
    queries = evaluation_surface(obj, start)
    total, var = gp_prefix(points, LENGTHS_M[0], NOISE_VARIANCE, queries, {initial, *indices.values()})
    baseline = total[initial]
    snapshots = {}
    for label, count in indices.items():
        snapshots[label] = dict(new_cells=count-initial, nats=float(total[count]-baseline),
                                remaining_variance=var[count], variance_reduction_fraction=1.-var[count]/var[initial])
    for s in TIMES:
        tick = round(s/DT)
        item = snapshots[f"t{s}"]
        q = next(row for row in quality["snapshots"] if row["s"] == s)
        if q["cells"] != item["new_cells"] + initial:
            raise ValueError("The information and coverage scorers selected different cells")
        item.update(wrist_translation_m=float(distance[tick]), wrist_rotation_rad=float(angle[tick]),
                    coverage=q["coverage"], fscore=q["signed"]["fscore"], light_three_s=q["low3_s"])
    sensitivity = {}
    for length in LENGTHS_M[1:]:
        alt, _ = gp_prefix(points, length, NOISE_VARIANCE)
        sensitivity[str(length)] = {label:float(alt[count]-alt[initial]) for label,count in indices.items()}
    counts = np.searchsorted(ticks, np.arange(round(SECONDS/DT)+1), side="right")
    np.savez_compressed(output/"information-curve.npz", increment_tick=ticks[initial:],
                        increment_nats=np.diff(total)[initial:], cumulative_nats=total[counts]-baseline,
                        new_cells=counts-initial, translation_m=distance, rotation_rad=angle)
    info = dict(obj=obj, start=start, arm=arm, seed=seed, spec=spec, initial_points=initial_points,
                initial_cells=initial, initial_query_variance=var[initial], snapshots=snapshots,
                sensitivity=sensitivity, first_new_cell_budget_s={
                    str(n):(float(ticks[initial+n-1]*DT) if initial+n<=len(ticks) else None) for n in CONTACT_BUDGETS})
    save_json(saved, info)
    print(f"{family} {obj}/{start} {arm}: {snapshots['t240']['nats']:.2f} nat", flush=True)


def run_native(root):
    for case in jobs():
        if case["series"] != "runs" or case["arm"] != "sweep-dp":
            continue
        obj, start = case["obj"], case["start"]
        key = Path(obj)/start
        wrist = OUT/"force-priority/runs"/key/"sweep-dp"
        mcc = root/"runs"/key/"sweep-mcc"
        arms = [("wrist-dp", wrist, wrist), ("wrist-mcc", mcc, mcc)]
        if obj == "pear":
            for name in ("aude-mcc", "aude-single-all-mcc"):
                arms.append((name, ACTIVE2/name, OUT/"references"/key/name))
        else:
            ref = OUT/"strong-baseline"/key/"aude-mcc"
            arms.append(("aude-mcc", ref, ref))
        for arm, source, quality in arms:
            evaluate(root, "native", obj, start, arm, source, quality/"scores.json")


def run_matched(root):
    for seed in PEAR_SEEDS:
        for arm in METHODS:
            if seed != PEAR_SEEDS[0] and arm.endswith("-mcc"):
                continue
            path = Path("pear/active-02")/f"seed-{seed}"/arm
            source = root/"matched"/path
            dest = root/"evaluation/matched"/path
            dest.mkdir(exist_ok=True, parents=True)
            score(source, dest, "pear", "active-02", list(TIMES))
            evaluate(root, "matched", "pear", "active-02", arm, source, dest/"scores.json", seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("native", "matched"))
    parser.add_argument("--root", type=Path, default=RESULTS)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    (run_native if args.family == "native" else run_matched)(args.root)
