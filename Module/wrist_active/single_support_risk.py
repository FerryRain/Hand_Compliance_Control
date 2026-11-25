"""Estimate loss of the last pad before another pad arrives, for a proposed palm motion."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

from . import policy_reach as reach


NEIGHBORS = 32
MAX_RISK = .30


def first_outcome(future, owner):
    """Classify each 100 Hz window by its first event. A direct hand-off counts as arrival."""
    zero = ~future.any(axis=-1)
    another = (future & (np.arange(4) != np.asarray(owner)[..., None, None])).any(axis=-1)
    length = future.shape[-2]
    loss_at = np.where(zero.any(axis=-1), zero.argmax(axis=-1), length)
    arrival_at = np.where(another.any(axis=-1), another.argmax(axis=-1), length)
    return np.where(loss_at < arrival_at, 1, np.where(arrival_at < length, 2, 0))


class RiskIndex:
    def __init__(self, features, actions, owners, losses):
        self.groups = {}
        for owner in range(4):
            keep = owners == owner
            if not keep.any():
                raise ValueError(f"No training windows for pad {owner}")
            for h, horizon in enumerate(reach.HORIZONS):
                values = np.c_[features[keep], reach.action_features(actions[keep, h], horizon)]
                self.groups[owner, h] = cKDTree(values), losses[keep, h]

    def query(self, features, actions, owner, h):
        states = np.broadcast_to(features, (len(actions), features.shape[-1]))
        query = np.c_[states, reach.action_features(actions, reach.HORIZONS[h])]
        tree, labels = self.groups[owner, h]
        distance, near = tree.query(query, k=np.arange(1, min(NEIGHBORS, tree.n) + 1))
        weights = np.exp(-.5 * (distance**2 - distance[:, :1]**2))
        # One positive and one negative pseudo-observation stabilize the empirical frequency.
        return (1. + (weights * labels[near]).sum(axis=1)) / (2. + weights.sum(axis=1))


class SingleSupportRisk:
    max_risk = MAX_RISK

    def __init__(self, path, reach_path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(data["metadata"].item())
            if (meta["reach_sha256"] != reach.file_hash(reach_path) or meta["version"] != 1
                    or meta["horizons_s"] != reach.HORIZONS.tolist() or meta["neighbors"] != NEIGHBORS):
                raise ValueError("Single-support predictor does not match the deployed reach model")
            self.index = RiskIndex(data["features"], data["actions"], data["owners"], data["losses"])

    def predict(self, joints, hits, local_normals, actions, h):
        owners = np.flatnonzero(hits)
        if len(owners) != 1:
            raise ValueError("A single-support risk query requires exactly one measured pad")
        return self.index.query(reach.state_features(joints, hits, local_normals), actions, int(owners[0]), h)


def training_labels(folder, record, expected):
    chunks, contacts = folder / "trace_chunks.json", folder / "live_head_contacts.npz"
    if (reach.file_hash(chunks) != record["chunks_sha256"]
            or reach.file_hash(contacts) != record["contacts_sha256"]):
        raise ValueError(f"Training receipt changed at {folder}")
    count = sum(chunk["states"] for chunk in json.loads(chunks.read_bytes()))
    hits = np.zeros((count, 4), dtype=bool)
    with np.load(contacts, allow_pickle=False) as data:
        valid = (data["max_region_load_n"] < 8.) & (data["normal_force_n"] >= .2)
        hits[data["step"][valid] - 1, data["finger"][valid]] = True
    # Use the exact start windows and ordering of the frozen policy reach database.
    dt = .002
    starts = np.arange(0, count - round(max(reach.HORIZONS) / dt), round(.20 / dt))
    starts = starts[hits[starts].any(axis=1)]
    np.testing.assert_array_equal(hits[starts], expected)
    only = expected.sum(axis=1) == 1
    owners = np.argmax(expected[only], axis=1)
    outcomes = []
    for horizon in reach.HORIZONS:
        future = hits[starts[only, None] + 5 * np.arange(1, round(horizon * 100) + 1)]
        outcomes.append(first_outcome(future, owners))
    return only, owners, np.stack(outcomes, axis=1)


def validate(features, actions, owners, loss, groups):
    """Hold out entire historical episodes, including all overlapping windows."""
    predicted, prior = np.zeros(loss.shape), np.zeros(loss.shape)
    for group in np.unique(groups):
        train = groups != group
        index = RiskIndex(features[train], actions[train], owners[train], loss[train])
        for owner in range(4):
            test = ~train & (owners == owner)
            if not test.any():
                continue
            background = loss[train & (owners == owner)]
            for h, _ in enumerate(reach.HORIZONS):
                predicted[test, h] = index.query(features[test], actions[test, h], owner, h)
                prior[test, h] = (1. + background[:, h].sum()) / (2. + len(background))
    return {str(horizon): dict(n=len(owners), loss_fraction=float(loss[:, h].mean()),
                predicted_mean=float(predicted[:, h].mean()),
                brier=float(np.mean((predicted[:, h] - loss[:, h])**2)),
                owner_prior_brier=float(np.mean((prior[:, h] - loss[:, h])**2)),
                auc=float(roc_auc_score(loss[:, h], predicted[:, h])))
            for h, horizon in enumerate(reach.HORIZONS)}


def fit(output, reach_path=reach.MODEL):
    with np.load(reach_path, allow_pickle=False) as data:
        meta = json.loads(data["metadata"].item())
        hits, features, actions = data["hits"], data["features"], data["actions"]
    if meta["version"] != 2 or meta["checkpoint"] != json.loads(reach.PROTOCOL.read_bytes())["checkpoint"]:
        raise ValueError("Unexpected reach database or finger policy")
    windows, owners, outcomes, groups = [], [], [], []
    offset = 0
    for group, (name, record) in enumerate(meta["episodes"].items()):
        end = offset + record["windows"]
        only, owner, classified = training_labels(reach.MODULE / "generated" / name, record, hits[offset:end])
        windows.append(offset + np.flatnonzero(only))
        owners.append(owner)
        outcomes.append(classified)
        groups.append(np.full(len(owner), group))
        offset = end
    if offset != len(hits):
        raise ValueError("Incomplete reach training history")
    windows, owners, outcomes, groups = map(np.concatenate, (windows, owners, outcomes, groups))
    losses = (outcomes == 1).astype(np.int8)
    x, u = features[windows], actions[windows]
    metadata = dict(version=1, reach_sha256=reach.file_hash(reach_path), checkpoint=meta["checkpoint"],
                    episodes=list(meta["episodes"]), horizons_s=reach.HORIZONS.tolist(), neighbors=NEIGHBORS,
                    source_windows=len(hits), single_windows=len(windows),
                    owner_counts=np.bincount(owners, minlength=4).tolist(),
                    held_out_episodes=validate(x, u, owners, losses, groups))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp.npz")
    np.savez_compressed(temp, metadata=json.dumps(metadata), features=x, actions=u, owners=owners,
                        losses=losses, outcomes=outcomes, episode=groups)
    temp.replace(output)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fit(args.out), indent=2))
